from src.chat.rp_agent import InferenceAgent
from src.config import config, logger

def main():
    try:
        inference_agent = InferenceAgent()
        final_response = inference_agent._agentic_flow("Who is John Doe according to context?")
        logger.info(f"Final response: {final_response}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise Exception(f"Error in main: {e}")
        

if __name__ == "__main__":
    main()