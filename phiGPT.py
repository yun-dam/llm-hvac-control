import pandas as pd
import numpy as np
import joblib
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DataFrameLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from eppy.modeleditor import IDF
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class phiGPT:
    def __init__(self, data_path, model_url="http://localhost:11434/", model_name="llama3.2:3b"):
        """Initialize the model, retriever, and simulation components."""
        self.model = ChatOllama(model=model_name, base_url=model_url)

        self.data_path = data_path

        # Load trained temperature prediction model and scaler
        self.temp_model = joblib.load('final_gbr_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        
        # Load the HVAC dataset
        self.full_df = pd.read_csv(self.data_path, parse_dates=["timestamp"])

        # Remove the last 144 hours for testing
        self.train_df = self.full_df.iloc[:-144]  # Training dataset
        self.test_df = self.full_df.iloc[-144:]   # Test dataset

        # Convert training data into 48-hour time series chunks
        self.time_series_chunks = self._create_time_series_chunks(self.train_df)

        # Generate embeddings & create FAISS vector store for training data
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        documents = [
            Document(
                page_content=str(chunk["data"]),
                metadata={"start_time": chunk["start_time"]}
            )
            for chunk in self.time_series_chunks
        ]
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.vectorstore.save_local("hvac_vector_db")
        
        # Set up retriever
        self.retriever = self.vectorstore.as_retriever()

        # Memory for chat history
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Define retrieval-augmented generation (RAG) chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.model, retriever=self.retriever, memory=self.memory
        )

    def _create_time_series_chunks(self, df, window=48, step=1):
        """Create rolling 48-hour time-series chunks from the dataset."""
        chunks = []
        for i in range(0, len(df) - window, step):
            chunk_data = df.iloc[i:i + window][["indoor_air_temperature", "cooling_set_point"]].values.tolist()
            chunks.append({
                "data": chunk_data,
                "start_time": df.iloc[i]["timestamp"]
            })
        return chunks

    def predict_temperature(self, input_features):
        """Predict indoor temperature based on input sensor data."""
        scaled_features = self.scaler.transform([input_features])
        return self.temp_model.predict(scaled_features)[0]


    def _find_similar_time_series(self, current_state):
        """Find the 10 most similar historical 48-hour sequences using DTW."""
        current_vector = np.array(current_state)
        distances = []

        for chunk in self.time_series_chunks:
            chunk_vector = np.array(chunk["data"])
            distance, _ = fastdtw(current_vector, chunk_vector, dist=euclidean)
            distances.append((distance, chunk["data"], chunk["start_time"]))

        # Sort by similarity (smallest distance first)
        distances.sort(key=lambda x: x[0])
        return [x[1] for x in distances[:10]]  # Return 10 closest time series
    
    def generate_optimized_setpoint(self, current_48h_series, input_features):
        """Retrieve similar historical sequences and generate HVAC control strategy."""
        similar_sequences = self._find_similar_time_series(current_48h_series)
        predicted_temp = self.predict_temperature(input_features)

        prompt_text = f"""
        You are an AI-powered HVAC optimization assistant. Your task is to analyze historical data and current conditions to recommend an optimal cooling set-point for the next hour.
        
        Context:
        - All temperature units are in Fahrenheit.
        - The goal is to prevent overcooling while maintaining occupant comfort.
        - You have access to ten historical 48-hour time series that are most similar to the current state.
        - Each time series contains hourly [indoor_air_temperature, cooling_set_point] pairs.
        - The predicted indoor temperature for the next hour is {predicted_temp:.2f}°F.
        
        Historical Data:
        {similar_sequences}
        
        Current 48-hour sequence:
        {current_48h_series}
        
        Instructions:
        1. Analyze the historical sequences and the current sequence.
        2. Consider the predicted indoor temperature.
        3. Determine the optimal cooling set-point for the next hour. Set-point should be one of 74, 76, 78 Fahrenheit.
        4. Provide a brief rationale for your recommendation (1-2 sentences).
        5. Format your response as a valid Python dictionary.
        
        Example Response:
        {{'optimal_cooling_setpoint': 74, 'rationale': 'Based on historical patterns and predicted temperature, 75.5°F balances energy efficiency and comfort.'}}
        """

        response = self.model.invoke(prompt_text)
        try:
            result_dict = eval(response.content)
            if not isinstance(result_dict, dict) or 'optimal_cooling_setpoint' not in result_dict:
                raise ValueError("Response format incorrect")
            return result_dict
        except Exception as e:
            return {'error': f'Failed to parse response: {str(e)}', 'raw_response': response.content}

    def conduct_24h_control(self, initial_input_features):
        """Conduct 24-hour HVAC control loop."""
        control_results = []
        input_features = initial_input_features.copy()

        for hour in range(24):
            setpoint_response = self.generate_optimized_setpoint(self.test_df.iloc[:48][["indoor_air_temperature", "cooling_set_point"]].values.tolist(), input_features)
            if 'optimal_cooling_setpoint' in setpoint_response:
                input_features[0] = setpoint_response['optimal_cooling_setpoint']
                predicted_temp = self.predict_temperature(input_features)
                control_results.append({'hour': hour + 1, 'setpoint': input_features[0], 'predicted_temp': predicted_temp})

        return control_results
    

# hvac_system = phiGPT('./vav_2-356.csv')

# test_df = hvac_system.test_df.iloc[:48]
# initial_input_features = test_df.iloc[48,1:-1].tolist() 

# control_results = hvac_system.conduct_24h_control(initial_input_features)
 

# ## 

# # Load the HVAC dataset
# full_df = pd.read_csv("indoor_temp_prediction.csv", parse_dates=["timestamp"])

# full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
# full_df['hour'] = full_df['timestamp'].dt.hour
# full_df['day_of_week'] = full_df['timestamp'].dt.dayofweek
# full_df = full_df.sort_values(by='timestamp')

# # Create lag features for previous 3 hours
# for lag in range(1, 4):
#     full_df[f'lag_{lag}'] = full_df['VAV 2-356'].shift(lag)

# # Drop rows with NaN values after creating lag features
# full_df.dropna(inplace=True)

# full_df[['timestamp','scheduled_sp', 'temperature', 'RH', 'hour', 'day_of_week', 'lag_1', 'lag_2', 'lag_3', 'VAV 2-356']].to_csv("vav_2_356_processed.csv", index=False)


