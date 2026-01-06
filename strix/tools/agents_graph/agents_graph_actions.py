import logging
import os
import threading
import uuid
from collections import deque
from datetime import UTC, datetime
from typing import Any, Literal

from strix.tools.registry import register_tool


logger = logging.getLogger(__name__)

# Agent concurrency control configuration
# Set via environment variables or defaults
MAX_CONCURRENT_AGENTS = int(os.getenv("STRIX_MAX_CONCURRENT_AGENTS", "3"))
AGENT_SPAWN_DELAY = float(os.getenv("STRIX_AGENT_SPAWN_DELAY", "2.0"))  # seconds between spawns

# Track actual running child agents (not root)
_running_child_count = 0
_running_child_lock = threading.Lock()

# Queue for pending agents waiting to be spawned
_pending_agents: deque[dict[str, Any]] = deque()
_pending_lock = threading.Lock()

_agent_graph: dict[str, Any] = {
    "nodes": {},
    "edges": [],
}

_root_agent_id: str | None = None

_agent_messages: dict[str, list[dict[str, Any]]] = {}

_running_agents: dict[str, threading.Thread] = {}

_agent_instances: dict[str, Any] = {}

_agent_states: dict[str, Any] = {}


def _summarize_context_with_llm(messages: list[dict[str, Any]]) -> str:
    """Use LLM to intelligently summarize parent's conversation for child agent.
    
    This creates a focused briefing containing:
    - Vulnerabilities and attack vectors discovered
    - Key endpoints, URLs, and attack surface
    - Credentials, tokens, or secrets found
    - Current progress and what's been tried
    - Dead ends to avoid duplicating work
    """
    if not messages:
        return ""
    
    # Build conversation text from all messages
    conversation_parts = []
    
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            if isinstance(content, list):
                # Handle multimodal content
                text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                content = "\n".join(text_parts)
            else:
                continue
        
        role = msg.get("role", "unknown")
        msg_text = f"[{role}]: {content}\n"
        conversation_parts.append(msg_text)
    
    if not conversation_parts:
        return ""
    
    conversation_text = "".join(conversation_parts)
    
    # Use LLM to summarize
    try:
        import litellm
        
        # Use the same model as agents - no hardcoded fallbacks
        model = os.getenv("STRIX_LLM")
        if not model:
            logger.warning("STRIX_LLM not set, cannot summarize context")
            return ""
        
        summary_prompt = f"""You are summarizing a security assessment conversation for a child agent.

Extract and condense ONLY the actionable intelligence into XML format:

<context_summary>
    <vulnerabilities>
        <!-- List confirmed/suspected vulnerabilities with proof -->
    </vulnerabilities>
    <attack_surface>
        <!-- Key endpoints, APIs, URLs discovered -->
    </attack_surface>
    <credentials>
        <!-- Any tokens, passwords, API keys, secrets found -->
    </credentials>
    <progress>
        <!-- What has been tested, current approach -->
    </progress>
    <dead_ends>
        <!-- Failed attempts to avoid repeating -->
    </dead_ends>
</context_summary>

Be concise but preserve critical technical details (exact URLs, parameters, payloads).
Do NOT include raw tool output - summarize the findings.
Output ONLY the XML, no markdown or other formatting.

CONVERSATION:
{conversation_text}

XML SUMMARY:"""

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": summary_prompt}],
            timeout=30,
        )
        
        summary = response.choices[0].message.content or ""
        logger.debug(f"LLM summarized {len(conversation_text)} chars to {len(summary)} chars")
        return summary.strip()
        
    except Exception as e:
        # If LLM fails for summarization, just pass empty context
        # The child agent will start fresh - better than garbage extraction
        # If the main LLM is down, the agent won't work anyway
        logger.warning(f"LLM summarization failed: {e}. Child agent will start with fresh context.")
        return ""


def _release_agent_slot() -> None:
    """Release a slot and process pending agents."""
    global _running_child_count
    
    with _running_child_lock:
        if _running_child_count > 0:
            _running_child_count -= 1
            logger.info(f"Released agent slot. Running: {_running_child_count}/{MAX_CONCURRENT_AGENTS}")
    
    _process_pending_agents()


def _process_pending_agents() -> None:
    """Process the pending agent queue, spawning agents as slots become available."""
    global _running_child_count
    import time
    
    from strix.agents import StrixAgent
    
    while True:
        pending_info = None
        current_count = 0
        
        # 1. Check capacity and pop from queue atomically
        # We do NOT hold the lock while sleeping or starting threads
        with _pending_lock:
            if not _pending_agents:
                break
                
            with _running_child_lock:
                if _running_child_count >= MAX_CONCURRENT_AGENTS:
                    logger.info(f"No agent slots available. Running: {_running_child_count}/{MAX_CONCURRENT_AGENTS}, Pending: {len(_pending_agents)}")
                    break
                
                # Acquire slot BEFORE starting agent
                _running_child_count += 1
                current_count = _running_child_count
            
            # Only pop if we successfully acquired a slot
            pending_info = _pending_agents.popleft()
            
        if not pending_info:
            break
            
        # 2. Process the popped agent outside the locks
        try:
            logger.info(f"Starting pending agent: {pending_info['name']} (Running: {current_count}/{MAX_CONCURRENT_AGENTS})")
            
            # Add delay between spawns to avoid overwhelming target
            if AGENT_SPAWN_DELAY > 0:
                time.sleep(AGENT_SPAWN_DELAY)
            
            # NOW create the StrixAgent - this adds it to graph with status="running"
            agent_config = pending_info["agent_config"]
            state = pending_info["state"]
            agent = StrixAgent(agent_config)
            _agent_instances[state.agent_id] = agent
            
            # Start the agent thread
            thread = threading.Thread(
                target=_run_agent_in_thread,
                args=(agent, state, pending_info["context_summary"]),
                daemon=True,
                name=f"Agent-{pending_info['name']}-{state.agent_id}",
            )
            thread.start()
            _running_agents[state.agent_id] = thread
            
        except Exception as e:
            logger.error(f"Failed to start pending agent {pending_info['name']}: {e}")
            # CRITICAL: Release the slot we reserved since we failed to start
            with _running_child_lock:
                _running_child_count -= 1
            
            # Mark as failed in graph if it was added
            state = pending_info["state"]
            if state.agent_id in _agent_graph["nodes"]:
                _agent_graph["nodes"][state.agent_id]["status"] = "failed"
                _agent_graph["nodes"][state.agent_id]["result"] = {"error": str(e)}
                _agent_graph["nodes"][state.agent_id]["finished_at"] = datetime.now(UTC).isoformat()


def _run_agent_in_thread(
    agent: Any, state: Any, context_summary: str
) -> dict[str, Any]:
    """Run an agent in a thread with compressed context instead of full history."""
    try:
        # Add compressed context summary from parent (if any)
        if context_summary:
            state.add_message("user", f"""<parent_context>
{context_summary}
</parent_context>""")

        # Add task assignment - identity is handled by LLM._build_identity_message()
        task_xml = f"""<task_assignment>
    <delegated_task>{state.task}</delegated_task>
    <parent_id>{state.parent_id}</parent_id>
</task_assignment>"""

        state.add_message("user", task_xml)

        _agent_states[state.agent_id] = state

        _agent_graph["nodes"][state.agent_id]["state"] = state.model_dump()

        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(agent.agent_loop(state.task))
        finally:
            loop.close()

    except Exception as e:
        _agent_graph["nodes"][state.agent_id]["status"] = "error"
        _agent_graph["nodes"][state.agent_id]["finished_at"] = datetime.now(UTC).isoformat()
        _agent_graph["nodes"][state.agent_id]["result"] = {"error": str(e)}
        _running_agents.pop(state.agent_id, None)
        _agent_instances.pop(state.agent_id, None)
        # Release slot for next pending agent
        _release_agent_slot()
        raise
    else:
        if state.stop_requested:
            _agent_graph["nodes"][state.agent_id]["status"] = "stopped"
        else:
            _agent_graph["nodes"][state.agent_id]["status"] = "completed"
        _agent_graph["nodes"][state.agent_id]["finished_at"] = datetime.now(UTC).isoformat()
        _agent_graph["nodes"][state.agent_id]["result"] = result
        _running_agents.pop(state.agent_id, None)
        _agent_instances.pop(state.agent_id, None)
        # Release slot for next pending agent
        _release_agent_slot()

        return {"result": result}


@register_tool(sandbox_execution=False)
def view_agent_graph(agent_state: Any) -> dict[str, Any]:
    try:
        structure_lines = ["=== AGENT GRAPH STRUCTURE ==="]

        def _build_tree(agent_id: str, depth: int = 0) -> None:
            node = _agent_graph["nodes"][agent_id]
            indent = "  " * depth

            you_indicator = " ← This is you" if agent_id == agent_state.agent_id else ""

            structure_lines.append(f"{indent}* {node['name']} ({agent_id}){you_indicator}")
            structure_lines.append(f"{indent}  Task: {node['task']}")
            structure_lines.append(f"{indent}  Status: {node['status']}")

            children = [
                edge["to"]
                for edge in _agent_graph["edges"]
                if edge["from"] == agent_id and edge["type"] == "delegation"
            ]

            if children:
                structure_lines.append(f"{indent}   Children:")
                for child_id in children:
                    _build_tree(child_id, depth + 2)

        root_agent_id = _root_agent_id
        if not root_agent_id and _agent_graph["nodes"]:
            for agent_id, node in _agent_graph["nodes"].items():
                if node.get("parent_id") is None:
                    root_agent_id = agent_id
                    break
            if not root_agent_id:
                root_agent_id = next(iter(_agent_graph["nodes"].keys()))

        if root_agent_id and root_agent_id in _agent_graph["nodes"]:
            _build_tree(root_agent_id)
        else:
            structure_lines.append("No agents in the graph yet")

        graph_structure = "\n".join(structure_lines)

        total_nodes = len(_agent_graph["nodes"])
        running_count = sum(
            1 for node in _agent_graph["nodes"].values() if node["status"] == "running"
        )
        waiting_count = sum(
            1 for node in _agent_graph["nodes"].values() if node["status"] == "waiting"
        )
        stopping_count = sum(
            1 for node in _agent_graph["nodes"].values() if node["status"] == "stopping"
        )
        completed_count = sum(
            1 for node in _agent_graph["nodes"].values() if node["status"] == "completed"
        )
        stopped_count = sum(
            1 for node in _agent_graph["nodes"].values() if node["status"] == "stopped"
        )
        failed_count = sum(
            1 for node in _agent_graph["nodes"].values() if node["status"] in ["failed", "error"]
        )
        queued_count = sum(
            1 for node in _agent_graph["nodes"].values() if node["status"] == "queued"
        )
        pending_in_queue = len(_pending_agents)

    except Exception as e:  # noqa: BLE001
        return {
            "error": f"Failed to view agent graph: {e}",
            "graph_structure": "Error retrieving graph structure",
        }
    else:
        with _running_child_lock:
            actual_running = _running_child_count
        return {
            "graph_structure": graph_structure,
            "concurrency_info": {
                "max_concurrent_agents": MAX_CONCURRENT_AGENTS,
                "currently_running": actual_running,
                "pending_in_queue": pending_in_queue,
            },
            "summary": {
                "total_agents": total_nodes,
                "running": running_count,
                "queued": queued_count,
                "waiting": waiting_count,
                "stopping": stopping_count,
                "completed": completed_count,
                "stopped": stopped_count,
                "failed": failed_count,
            },
        }


@register_tool(sandbox_execution=False)
def create_agent(
    agent_state: Any,
    task: str,
    name: str | None = None,
    inherit_context: bool = True,
    prompt_modules: str | None = None,
) -> dict[str, Any]:
    if name is None:
        name = f"Agent-{uuid.uuid4().hex[:6]}"

    try:
        # Enforce hierarchy: Only Root Agent (or agents with no parent) can spawn new agents directly.
        # Sub-agents must request spawns via agent_finish.
        if hasattr(agent_state, "parent_id") and agent_state.parent_id is not None:
             return {
                "success": False,
                "error": (
                    "PERMISSION DENIED: Sub-agents are NOT allowed to spawn new agents directly. "
                    "You must complete your current task and request new agents via the 'agent_finish' tool. "
                    "Use the 'spawn_requests' parameter in 'agent_finish' to specify the agents you want to create."
                ),
                "agent_id": None,
            }

        parent_id = agent_state.agent_id

        module_list = []
        if prompt_modules:
            module_list = [m.strip() for m in prompt_modules.split(",") if m.strip()]

        if len(module_list) > 5:
            return {
                "success": False,
                "error": (
                    "Cannot specify more than 5 prompt modules for an agent "
                    "(use comma-separated format)"
                ),
                "agent_id": None,
            }

        if module_list:
            from strix.prompts import get_all_module_names, validate_module_names

            validation = validate_module_names(module_list)
            if validation["invalid"]:
                available_modules = list(get_all_module_names())
                return {
                    "success": False,
                    "error": (
                        f"Invalid prompt modules: {validation['invalid']}. "
                        f"Available modules: {', '.join(available_modules)}"
                    ),
                    "agent_id": None,
                }

        from strix.agents.state import AgentState
        from strix.llm.config import LLMConfig
        import uuid

        # Prepare state and config BEFORE checking concurrency
        # DO NOT create StrixAgent yet - it adds to graph with status="running"
        if not name:
            name = f"Agent-{uuid.uuid4().hex[:6]}"

        state = AgentState(task=task, agent_name=name, parent_id=parent_id, max_iterations=300)

        parent_agent = _agent_instances.get(parent_id)

        timeout = None
        scan_mode = "deep"
        if parent_agent and hasattr(parent_agent, "llm_config"):
            if hasattr(parent_agent.llm_config, "timeout"):
                timeout = parent_agent.llm_config.timeout
            if hasattr(parent_agent.llm_config, "scan_mode"):
                scan_mode = parent_agent.llm_config.scan_mode

        llm_config = LLMConfig(prompt_modules=module_list, timeout=timeout, scan_mode=scan_mode)

        agent_config = {
            "llm_config": llm_config,
            "state": state,
        }
        if parent_agent and hasattr(parent_agent, "non_interactive"):
            agent_config["non_interactive"] = parent_agent.non_interactive

        # Summarize context using LLM BEFORE deciding to queue or start
        context_summary = ""
        if inherit_context:
            parent_messages = agent_state.get_conversation_history()
            context_summary = _summarize_context_with_llm(parent_messages)
            logger.debug(f"Summarized {len(parent_messages)} messages to {len(context_summary)} chars for child agent")

        # Check if we have capacity to start immediately
        global _running_child_count
        
        can_start = False
        current_count = 0
        
        with _running_child_lock:
            if _running_child_count < MAX_CONCURRENT_AGENTS:
                _running_child_count += 1
                can_start = True
            current_count = _running_child_count
        
        if can_start:
            try:
                # Now create and start the agent - this adds it to graph with status="running"
                from strix.agents import StrixAgent
                agent = StrixAgent(agent_config)
                _agent_instances[state.agent_id] = agent
                
                logger.info(f"Starting agent '{name}' immediately (Running: {current_count}/{MAX_CONCURRENT_AGENTS})")
                thread = threading.Thread(
                    target=_run_agent_in_thread,
                    args=(agent, state, context_summary),
                    daemon=True,
                    name=f"Agent-{name}-{state.agent_id}",
                )
                thread.start()
                _running_agents[state.agent_id] = thread
                spawn_status = "started"
            except Exception as e:
                # CRITICAL: Release the slot if startup fails
                with _running_child_lock:
                    _running_child_count -= 1
                
                # Mark as failed in graph if it was added
                if state.agent_id in _agent_graph["nodes"]:
                    _agent_graph["nodes"][state.agent_id]["status"] = "failed"
                    _agent_graph["nodes"][state.agent_id]["result"] = {"error": str(e)}
                    _agent_graph["nodes"][state.agent_id]["finished_at"] = datetime.now(UTC).isoformat()
                
                raise
        else:
            # Queue the agent config for later - do NOT create StrixAgent yet
            pending_count = len(_pending_agents)
            logger.info(f"Queueing agent '{name}' (Running: {current_count}/{MAX_CONCURRENT_AGENTS}, Pending: {pending_count})")
            with _pending_lock:
                _pending_agents.append({
                    "agent_config": agent_config,  # Config, not instance
                    "state": state,
                    "name": name,
                    "context_summary": context_summary,
                })
            
            # Add placeholder to graph so it's visible in UI immediately
            _agent_graph["nodes"][state.agent_id] = {
                "id": state.agent_id,
                "name": name,
                "task": task,
                "status": "queued",
                "parent_id": parent_id,
                "created_at": datetime.now(UTC).isoformat(),
                "finished_at": None,
                "result": None,
                "llm_config": "pending",
                "agent_type": "StrixAgent",
                "state": state.model_dump(),
            }
            
            # Double check if we can process now (race condition fix)
            # A slot might have opened up between our check and the append
            _process_pending_agents()
            
            spawn_status = "queued"

    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"Failed to create agent: {e}", "agent_id": None}
    else:
        return {
            "success": True,
            "agent_id": state.agent_id,
            "message": f"Agent '{name}' {spawn_status} (max {MAX_CONCURRENT_AGENTS} concurrent agents)",
            "spawn_status": spawn_status,
            "pending_count": len(_pending_agents),
            "agent_info": {
                "id": state.agent_id,
                "name": name,
                "status": "running" if spawn_status == "started" else "queued",
                "parent_id": parent_id,
            },
        }


@register_tool(sandbox_execution=False)
def send_message_to_agent(
    agent_state: Any,
    target_agent_id: str,
    message: str,
    message_type: Literal["query", "instruction", "information"] = "information",
    priority: Literal["low", "normal", "high", "urgent"] = "normal",
) -> dict[str, Any]:
    try:
        if target_agent_id not in _agent_graph["nodes"]:
            return {
                "success": False,
                "error": f"Target agent '{target_agent_id}' not found in graph",
                "message_id": None,
            }

        sender_id = agent_state.agent_id

        from uuid import uuid4

        message_id = f"msg_{uuid4().hex[:8]}"
        message_data = {
            "id": message_id,
            "from": sender_id,
            "to": target_agent_id,
            "content": message,
            "message_type": message_type,
            "priority": priority,
            "timestamp": datetime.now(UTC).isoformat(),
            "delivered": False,
            "read": False,
        }

        if target_agent_id not in _agent_messages:
            _agent_messages[target_agent_id] = []

        _agent_messages[target_agent_id].append(message_data)

        _agent_graph["edges"].append(
            {
                "from": sender_id,
                "to": target_agent_id,
                "type": "message",
                "message_id": message_id,
                "message_type": message_type,
                "priority": priority,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )

        message_data["delivered"] = True

        target_name = _agent_graph["nodes"][target_agent_id]["name"]
        sender_name = _agent_graph["nodes"][sender_id]["name"]

        return {
            "success": True,
            "message_id": message_id,
            "message": f"Message sent from '{sender_name}' to '{target_name}'",
            "delivery_status": "delivered",
            "target_agent": {
                "id": target_agent_id,
                "name": target_name,
                "status": _agent_graph["nodes"][target_agent_id]["status"],
            },
        }

    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"Failed to send message: {e}", "message_id": None}


@register_tool(sandbox_execution=False)
def agent_finish(
    agent_state: Any,
    result_summary: str,
    findings: list[str] | None = None,
    success: bool = True,
    report_to_parent: bool = True,
    final_recommendations: list[str] | None = None,
    spawn_requests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Complete agent execution and report summarized findings back to parent.
    
    CRITICAL: You must SUMMARIZE your findings, not dump raw tool output!
    
    Your job is to distill your work into actionable intelligence:
    
    result_summary: A concise paragraph explaining what you found and accomplished.
                   Focus on the key takeaways, not a play-by-play.
    
    findings: List of discrete, actionable findings. Each should be:
              - One specific item (vulnerability, endpoint, credential, etc.)
              - Include proof/evidence inline (URL, parameter, payload)
              - Summarized, not raw output
              
              BAD:  "nmap output: PORT STATE SERVICE VERSION\\n22/tcp open ssh..."
              GOOD: "OpenSSH 8.2p1 on port 22 - vulnerable to CVE-2020-15778"
              
              BAD:  "Found the following endpoints: /api/users, /api/admin, /api/..."
              GOOD: "Unauthenticated admin API at /api/admin/users returns user list"
    
    final_recommendations: Prioritized next steps for parent/other agents.

    spawn_requests: List of agents you recommend spawning to investigate findings further.
                    Each request should be a dictionary with:
                    - name: str (Name of the agent)
                    - task: str (Detailed task description)
                    - prompt_modules: str (Optional, comma-separated list of modules)
                    
                    Example:
                    [
                        {
                            "name": "NPM Vulnerability Agent",
                            "task": "Investigate Nginx Proxy Manager on port 81...",
                            "prompt_modules": "web_vuln_scanner"
                        }
                    ]
    """
    try:
        if not hasattr(agent_state, "parent_id") or agent_state.parent_id is None:
            return {
                "agent_completed": False,
                "error": (
                    "This tool can only be used by subagents. "
                    "Root/main agents must use finish_scan instead."
                ),
                "parent_notified": False,
            }

        agent_id = agent_state.agent_id

        if agent_id not in _agent_graph["nodes"]:
            return {"agent_completed": False, "error": "Current agent not found in graph"}

        agent_node = _agent_graph["nodes"][agent_id]

        # Store findings as-is (agent is responsible for summarizing)
        agent_node["status"] = "finished" if success else "failed"
        agent_node["finished_at"] = datetime.now(UTC).isoformat()
        agent_node["result"] = {
            "summary": result_summary,
            "findings": findings or [],
            "success": success,
            "recommendations": final_recommendations or [],
            "spawn_requests": spawn_requests or [],
        }

        parent_notified = False

        if report_to_parent and agent_node["parent_id"]:
            parent_id = agent_node["parent_id"]

            if parent_id in _agent_graph["nodes"]:
                # Build findings XML
                findings_xml = "\n".join(
                    f"        <finding>{finding}</finding>" for finding in (findings or [])
                )
                recommendations_xml = "\n".join(
                    f"        <recommendation>{rec}</recommendation>"
                    for rec in (final_recommendations or [])
                )
                
                spawn_requests_xml = ""
                if spawn_requests:
                    spawn_requests_xml = "        <spawn_requests>\n"
                    for req in spawn_requests:
                        spawn_requests_xml += "            <request>\n"
                        spawn_requests_xml += f"                <name>{req.get('name', 'Unknown')}</name>\n"
                        spawn_requests_xml += f"                <task>{req.get('task', '')}</task>\n"
                        if req.get('prompt_modules'):
                            spawn_requests_xml += f"                <prompt_modules>{req.get('prompt_modules')}</prompt_modules>\n"
                        spawn_requests_xml += "            </request>\n"
                    spawn_requests_xml += "        </spawn_requests>"

                report_message = f"""<agent_completion_report>
    <agent_info>
        <agent_name>{agent_node["name"]}</agent_name>
        <agent_id>{agent_id}</agent_id>
        <task>{agent_node["task"]}</task>
        <status>{"SUCCESS" if success else "FAILED"}</status>
    </agent_info>
    <results>
        <summary>{result_summary}</summary>
        <findings>
{findings_xml}
        </findings>
        <recommendations>
{recommendations_xml}
        </recommendations>
{spawn_requests_xml}
    </results>
</agent_completion_report>"""

                if parent_id not in _agent_messages:
                    _agent_messages[parent_id] = []

                from uuid import uuid4

                _agent_messages[parent_id].append(
                    {
                        "id": f"report_{uuid4().hex[:8]}",
                        "from": agent_id,
                        "to": parent_id,
                        "content": report_message,
                        "message_type": "information",
                        "priority": "high",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "delivered": True,
                        "read": False,
                    }
                )

                parent_notified = True

        _running_agents.pop(agent_id, None)

        return {
            "agent_completed": True,
            "parent_notified": parent_notified,
            "completion_summary": {
                "agent_id": agent_id,
                "agent_name": agent_node["name"],
                "task": agent_node["task"],
                "success": success,
                "findings_count": len(findings or []),
                "has_recommendations": bool(final_recommendations),
                "finished_at": agent_node["finished_at"],
            },
        }

    except Exception as e:  # noqa: BLE001
        return {
            "agent_completed": False,
            "error": f"Failed to complete agent: {e}",
            "parent_notified": False,
        }


def stop_agent(agent_id: str) -> dict[str, Any]:
    try:
        if agent_id not in _agent_graph["nodes"]:
            return {
                "success": False,
                "error": f"Agent '{agent_id}' not found in graph",
                "agent_id": agent_id,
            }

        agent_node = _agent_graph["nodes"][agent_id]

        if agent_node["status"] in ["completed", "error", "failed", "stopped"]:
            return {
                "success": True,
                "message": f"Agent '{agent_node['name']}' was already stopped",
                "agent_id": agent_id,
                "previous_status": agent_node["status"],
            }

        if agent_id in _agent_states:
            agent_state = _agent_states[agent_id]
            agent_state.request_stop()

        if agent_id in _agent_instances:
            agent_instance = _agent_instances[agent_id]
            if hasattr(agent_instance, "state"):
                agent_instance.state.request_stop()
            if hasattr(agent_instance, "cancel_current_execution"):
                agent_instance.cancel_current_execution()

        agent_node["status"] = "stopping"

        try:
            from strix.telemetry.tracer import get_global_tracer

            tracer = get_global_tracer()
            if tracer:
                tracer.update_agent_status(agent_id, "stopping")
        except (ImportError, AttributeError):
            pass

        agent_node["result"] = {
            "summary": "Agent stop requested by user",
            "success": False,
            "stopped_by_user": True,
        }

        return {
            "success": True,
            "message": f"Stop request sent to agent '{agent_node['name']}'",
            "agent_id": agent_id,
            "agent_name": agent_node["name"],
            "note": "Agent will stop gracefully after current iteration",
        }

    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "error": f"Failed to stop agent: {e}",
            "agent_id": agent_id,
        }


def send_user_message_to_agent(agent_id: str, message: str) -> dict[str, Any]:
    try:
        if agent_id not in _agent_graph["nodes"]:
            return {
                "success": False,
                "error": f"Agent '{agent_id}' not found in graph",
                "agent_id": agent_id,
            }

        agent_node = _agent_graph["nodes"][agent_id]

        if agent_id not in _agent_messages:
            _agent_messages[agent_id] = []

        from uuid import uuid4

        message_data = {
            "id": f"user_msg_{uuid4().hex[:8]}",
            "from": "user",
            "to": agent_id,
            "content": message,
            "message_type": "instruction",
            "priority": "high",
            "timestamp": datetime.now(UTC).isoformat(),
            "delivered": True,
            "read": False,
        }

        _agent_messages[agent_id].append(message_data)

        return {
            "success": True,
            "message": f"Message sent to agent '{agent_node['name']}'",
            "agent_id": agent_id,
            "agent_name": agent_node["name"],
        }

    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "error": f"Failed to send message to agent: {e}",
            "agent_id": agent_id,
        }


@register_tool(sandbox_execution=False)
def wait_for_message(
    agent_state: Any,
    reason: str = "Waiting for messages from other agents",
) -> dict[str, Any]:
    try:
        agent_id = agent_state.agent_id
        agent_name = agent_state.agent_name

        agent_state.enter_waiting_state()

        if agent_id in _agent_graph["nodes"]:
            _agent_graph["nodes"][agent_id]["status"] = "waiting"
            _agent_graph["nodes"][agent_id]["waiting_reason"] = reason

        try:
            from strix.telemetry.tracer import get_global_tracer

            tracer = get_global_tracer()
            if tracer:
                tracer.update_agent_status(agent_id, "waiting")
        except (ImportError, AttributeError):
            pass

    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"Failed to enter waiting state: {e}", "status": "error"}
    else:
        return {
            "success": True,
            "status": "waiting",
            "message": f"Agent '{agent_name}' is now waiting for messages",
            "reason": reason,
            "agent_info": {
                "id": agent_id,
                "name": agent_name,
                "status": "waiting",
            },
            "resume_conditions": [
                "Message from another agent",
                "Message from user",
                "Direct communication",
                "Waiting timeout reached",
            ],
        }


@register_tool(sandbox_execution=False)
def manage_agent(
    agent_id: str,
    action: Literal["terminate", "status"],
    reason: str | None = None,
) -> dict[str, Any]:
    if agent_id not in _agent_states:
        return {
            "success": False,
            "error": f"Agent with ID '{agent_id}' not found.",
            "status": "not_found",
        }

    state = _agent_states[agent_id]

    if action == "status":
        node_info = _agent_graph["nodes"].get(agent_id, {})
        return {
            "success": True,
            "agent_id": agent_id,
            "status": node_info.get("status", "unknown"),
            "task": state.task,
            "iteration": state.iteration,
            "last_updated": state.last_updated,
            "completed": state.completed,
            "stop_requested": state.stop_requested,
        }

    if action == "terminate":
        if state.completed:
            return {
                "success": False,
                "error": f"Agent '{agent_id}' is already completed.",
                "status": "completed",
            }

        state.request_stop()

        # Update graph status
        if agent_id in _agent_graph["nodes"]:
            _agent_graph["nodes"][agent_id]["status"] = "terminating"

        return {
            "success": True,
            "message": f"Termination requested for agent '{agent_id}'. It should stop shortly.",
            "status": "terminating",
        }

    return {"success": False, "error": f"Unknown action: {action}"}
