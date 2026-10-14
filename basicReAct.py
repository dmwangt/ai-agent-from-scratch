import re

# 1. 定义两个简单的工具
def calculate(expression):
    """计算数学表达式的值"""
    try:
        return str(eval(expression))
    except:
        return "错误：无法计算该表达式"

def web_search(query):
    """模拟网络搜索"""
    if "2026年奥运会" in query or "2026" in query:
        return "2026年冬季奥运会将在意大利米兰-皮埃普查举行。"
    return "未找到相关搜索结果。"

# 2. 模拟 LLM 的预测行为（根据不同的输入历史，返回特定的 ReAct 文本）
def mock_llm_response(execution_history):
    # 只读取实际执行的工具记录，不检查系统提示词中的格式示例。
    tool_results = [
        entry for entry in execution_history if entry["type"] == "tool_result"
    ]

    # 模拟第一轮：模型看到问题后，决定调用搜索工具
    if not tool_results:
        return (
            "Thought: 用户想知道2026年奥运会举办地和距今（假设2024年）有多少年。我需要先搜索2026年奥运会的举办地。\n"
            "Action: web_search[2026年奥运会]"
        )
    
    # 模拟第二轮：模型拿到了搜索结果，决定调用计算器计算年份差距
    last_result = tool_results[-1]
    if (
        last_result["tool_name"] == "web_search"
        and last_result["result"].startswith("2026年冬季奥运会")
    ):
        return (
            "Thought: 我现在知道2026年奥运会在意大利举办了。接下来我需要计算从2024年到2026年还有多少年：2026 - 2024。\n"
            "Action: calculate[2026 - 2024]"
        )
    
    # 模拟第三轮：模型拿到了计算结果，给出最终答案
    if last_result["tool_name"] == "calculate" and last_result["result"] == "2":
        return (
            "Thought: 我现在有了所有需要的信息。\n"
            "Final Answer: 2026年冬季奥运会将在意大利米兰举行，距离2024年还有 2 年。"
        )
        
    return "Thought: 我不确定下一步该做什么。\nFinal Answer: 抱歉，我无法回答这个问题。"

def run_react_agent(user_question):
    # 基础提示词，规范 LLM 的输出格式
    system_prompt = (
        "你是一个具备思考和行动能力的 AI 助手。请使用以下格式回答问题：\n"
        "Thought: 你的思考过程\n"
        "Action: 工具名称[工具参数]\n"
        "Observation: 工具返回的结果（由系统提供）\n"
        "Final Answer: 最终给用户的答案\n\n"
        f"问题: {user_question}\n"
    )
    
    # 系统提示词单独保留；真实 LLM 可同时接收它和执行历史。
    # 当前 mock 只需要执行历史来判断下一步。
    execution_history = []
    max_steps = 5  # 防止死循环
    
    print(f"🚀 收到用户问题: {user_question}\n" + "="*40)
    
    for step in range(max_steps):
        # 1. 让 LLM “思考”并输出下一步
        llm_output = mock_llm_response(execution_history)
        print(llm_output)
        
        # 将模型输出保存为独立记录
        execution_history.append({"type": "assistant", "content": llm_output})
        
        # 2. 检查模型是否给出了最终答案
        if "Final Answer:" in llm_output:
            print("="*40 + "\n✅ Agent 运行结束！")
            return
        
        # 3. 解析 Action：正则匹配 Action: 工具名[参数]
        action_match = re.search(r"Action:\s*(\w+)\[(.*)\]", llm_output)
        if action_match:
            tool_name = action_match.group(1)
            tool_arg = action_match.group(2)
            
            print(f"⚙️ [系统动作] 正在调用工具: {tool_name}({tool_arg})...")
            
            # 4. 执行对应的工具逻辑
            if tool_name == "web_search":
                observation = web_search(tool_arg)
            elif tool_name == "calculate":
                observation = calculate(tool_arg)
            else:
                observation = f"错误：未知的工具 {tool_name}"
                
            print(f"👁️ [系统观察] 工具返回: {observation}\n" + "-"*40)
            
            # 5. 保存结构化工具结果，供下一轮读取
            execution_history.append({
                "type": "tool_result",
                "tool_name": tool_name,
                "argument": tool_arg,
                "result": observation,
            })
        else:
            print("⚠️ 无法解析 Action，强制结束。")
            break

# 运行 Agent
if __name__ == "__main__":
    run_react_agent("2026年奥运会在哪里举行？距离2024年还有几年？")
