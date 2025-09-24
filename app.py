import yaml
import os
import io
import sys
import pandas as pd
import traceback
from util import constant
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt,wait_fixed

from llm.zhipu import format_messages, completion

load_dotenv()
MAX_LOOP = 3

"""
#"2025的总资产是多少？",
#"2018年的总营收是多少？",
#"2023收入同比增加了多少？",
#"相较于24年，25年的总负债增加了多少？",
#"2018到2025，该公司的营收复合增长率是多少？",
#"为什么22年总资产显著增加?",
#"相较于去年，2025的利润率提升的主要驱动力是什么?",
#"相较于其他年份，2021最大的异常是什么？",
#"该公司的Debt to Equity Ratio在过去五年中有何变化？这一变化反映了企业在不同阶段的杠杆策略有何特点？",
#"该公司对大额商誉减值的风险敞口有多大？此类减值将对权益和杠杆率产生什么影响？"     
"""
question = "相较于去年,2025的利润率提升的主要驱动力是什么?"
try:
    ##### Load Causal Graph and Sentiments #####
    # Classify user question with LLMs to determine which YAML to load
    CLASSIFY_SYS = "You are a helpful assistant that classifies financial analysis questions."
    CLASSIFY_USER = f"""
    请将##问题##分类为下述项目之一，并返回项目名称，不要解释。
    - balance_sheet: 适用于涉及资产负债表项目（如资产、负债、股东权益等）的分析问题。
    - income_statement: 适用于涉及损益表项目（如收入、 费用、利润等）的分析问题。
    - others: 适用于不涉及因果关系分析的问题，或问题过于宽泛，无法归类到上述两类中的任何一类。
    ##问题## 
    {question}
    """
    usr_msg = CLASSIFY_USER.format( question=question)
    classify_messages = format_messages(CLASSIFY_SYS, usr_msg )
    print(f"\n================= 分类问题: {question} =================\n")
    filename = completion(classify_messages, 
                            model=constant.GLM45FLASH, 
                            thinking_type="disabled", 
                            stream=False).strip()

    if filename not in ['balance_sheet', 'income_statement']:
        print(f"Question '{question}' is classified as '{filename}', which is not supported. Skipping.\n")

    yaml_filename = filename + '.yaml'
    csv_filename = filename + '.csv'
    print(f"1. 问题 '{question}' 被分类为 '{yaml_filename}'.")
    causal_yaml_path = os.path.join(os.path.dirname(__file__), 'data', yaml_filename)

    # Read YAML file from data folder
    with open(causal_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # Retrieve sections
    graph = yaml_data.get('causal_graph')
    sentiment = yaml_data.get('sentiments')

    annual_csv_path = os.path.join(os.path.dirname(__file__), 'data', csv_filename)
    annual_df = pd.read_csv(annual_csv_path)
    sample = annual_df.head(1)

    ##### Analyze Data and Generate Facts #####
    ANALYTICS_SYS = "You are a data analyst that good at data analysis and statistics"
    ANALYTICS_USER = """
    Based on the causal graph, sentiments among variables and data schema provided, please generate code to list facts which can help answer the following question. 
    ## REQUIREMENTS
    - load data from folder '{path}' based on the schema, sample and question provided 
    - use entire causal graph and all sentiments among variables, don't miss any important node or sentiment
    - analyze as deep as possible in causal graph to find the root cause
    - only output code
    - the code should be runnable in Python environment
    - DO NOT get to conclusions, only list facts you found
    - DO NOT use pd.StringIO function, use io.StringIO instead
    - DO NOT make up any data, read data from the path provided
    - Print facts you found in Chinese

    ## RULES
    - all data in data source are numeric, including year column


    ## Causal Graph 
    {graph}

    ## Sentiments 
    {sentiment} 

    ## Data Schema & Sample
    {sample}

    ## Question
    {question}
    """

    facts = None
    error_msg = None

    for i in range(MAX_LOOP):
        if i == 0:
            usr_msg = ANALYTICS_USER.format(
                graph=graph,
                sentiment=sentiment,
                path=annual_csv_path,
                sample=sample.to_string(index=False),
                question=question
            )
        else:
            # Regenerate code with error reflection
            REFLECT_USER = ANALYTICS_USER + f"\n\n# Error encountered in previous code execution:\n{error_msg}\n请根据上述错误信息修正代码并重新生成。"
            usr_msg = REFLECT_USER.format(
                graph=graph,
                sentiment=sentiment,
                path=annual_csv_path,
                sample=sample.to_string(index=False),
                question=question
            )

        messages = format_messages(ANALYTICS_SYS, usr_msg)
        print("\n================= 生成代码 =================\n")
        generated_code = completion(
            messages,
            model=constant.GLM45FLASH,
            stream=True,
            thinking_type="disabled"
        )
        generated_code = generated_code.replace("```python", "").replace("```", "").strip()
       
    
        stdout_buffer = io.StringIO()
        try:
            sys_stdout = sys.stdout
            sys.stdout = stdout_buffer
            print("\n================= 执行代码 =================\n")
            exec(generated_code)
            facts = stdout_buffer.getvalue()
            error_msg = None
            break
        except Exception as e:
            error_msg = f"{str(e)}\nTraceback:\n{traceback.format_exc()}"
            facts = f"Error: {error_msg}"
        finally:
            sys.stdout = sys_stdout
            print("\n================= 代码执行结果: Facts =================\n")
            print(facts)

    ##### Answering the Question with Facts #####
    ANSWERING_SYS = "You are a Causal Model that is good at answering 'Why' question with cause-effect and facts"
    ANSWERING_USER = """
    Please answer the following question based on the causal graph, sentiments among variables and facts provided:
    ## REQUIREMENTS
    - leveraging full causal graph and all sentiments among variables to find the root cause as deep as possible
    - use facts provided to support your answer
    - DONT make up any fact
    - Your output should be in Chinese

    ## Causal Graph 
    {graph}

    ## Sentiments 
    {sentiment} 

    ## Facts
    {facts}

    ## Question
    {question}
    """

    usr_msg = ANSWERING_USER.format(graph=graph, 
                            sentiment=sentiment, 
                            facts=facts,
                            question=question)
    messages = format_messages(ANSWERING_SYS,usr_msg)
    print("\n================= 进行分析 =================\n")
    answer = completion(messages, 
                        model=constant.GLM45FLASH,
                        thinking_type="enabled",
                        stream=True
                        )
    print("\n================= 最终结论 =================\n")
    print(answer)
except Exception as e:
    print(f"\nException occurred while processing question '{question}': {e}\nTraceback:\n{traceback.format_exc()}\nContinuing to next question...\n")


