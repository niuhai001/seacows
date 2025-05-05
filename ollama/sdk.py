from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

SPARK_URL="wss://spark-api.xf-yun.com/v4.0/chat"
SPARK_DOMAIN="generalv4.0"
SPARK_APP_ID="c31c86a6"
SPARK_API_KEY="982b899ec4da15cbe2156af77a536430"
SPARK_API_SECRET="YjdkNDQyMjllZDRjM2U0NDJjY2E4NGM5"

if __name__ == "__main__":
    # Initialize the chat model
    llm = ChatSparkLLM(
        spark_api_url=SPARK_URL,
        spark_llm_domain=SPARK_DOMAIN,
        spark_app_id=SPARK_APP_ID,
        api_key=SPARK_API_KEY,
        api_secret=SPARK_API_SECRET,
        streaming=False,
    )
    
    # Create a chat message
    message = ChatMessage(role="user", content="你好，姜总")
    
    # Get the response from the model
    handler = ChunkPrintHandler()
    a = llm.generate([message], callable=[handler])
    
    # Print the response
    print(a)