import json

from modules.llm import LLMFactory
from modules.chat_engine import ChatEngine


base_llm = LLMFactory.create("Ollama")
llm = base_llm(llm_name="llama3.1", base_url="http://localhost:11434")

def test():
    query = "When does this picture appear in the video?"
    context = json.dumps({'video_name': 'Top 10 Beautiful Places to Visit in Sweden', 'content': '{"video_name": "Top 10 Beautiful Places to Visit in Sweden", "text": "a river in the middle of a snowy forest", "img_path": "/home/quangvodc/video-chatbot/video-understanding-chatbot/data/frames/Top 10 Beautiful Places to Visit in Sweden/00001.jpg", "start_time": "0:00:01", "end_time": "0:00:01"}', 'sim_score': 0.9969779253005981})

    chat_engine = ChatEngine(llm=llm)
    
    output = chat_engine.invoke(context=context, query=query)
    print(output)

    assert len(output) > 0