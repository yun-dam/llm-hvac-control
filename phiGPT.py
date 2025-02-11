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
    
    def generate_optimized_setpoint(self, current_48h_series):
        """Retrieve similar historical sequences and generate HVAC control strategy."""
        similar_sequences = self._find_similar_time_series(current_48h_series)
        # predicted_temp = self.predict_temperature(input_features)
        # - The predicted indoor temperature for the next hour is {predicted_temp:.2f}°F.
        prompt_text = f"""
        You are an AI-powered HVAC optimization assistant. Your task is to analyze historical data and current conditions to recommend an optimal cooling set-point for the next hour.
        
        Context:
        - All temperature units are in Fahrenheit.
        - The goal is to prevent overcooling while maintaining occupant comfort.
        - You have access to ten historical 48-hour time series that are most similar to the current state.
        - Each time series contains hourly [indoor_air_temperature, cooling_set_point] pairs.
        
        
        Historical Data:
        {similar_sequences}
        
        Current 48-hour sequence:
        {current_48h_series}
        
        Instructions:
        1. Analyze the historical sequences and the current sequence.
        2. Determine the optimal cooling set-point for the next hour. Set-point must be either 74, 76, or 78 Fahrenheit.
        3. Format your response in to the following format, which is a python dictionary:
        
        IMPORTANT: You must ONLY respond with the dictionary. Do not provide any additional text or explanations.

        FORMAT:
        {{'optimal_cooling_setpoint': "YOUR OPTIMAL SET-POINT (Int)"}}
        """

        response = self.model.invoke(prompt_text)
        try:
            result_dict = eval(response.content)
            if not isinstance(result_dict, dict) or 'optimal_cooling_setpoint' not in result_dict:
                raise ValueError("Response format incorrect")
            return result_dict
        except Exception as e:
            return {'error': f'Failed to parse response: {str(e)}', 'raw_response': response.content}

    # def conduct_24h_control(self, initial_input_features):
    #     """Conduct 24-hour HVAC control loop."""
    #     control_results = []
    #     input_features = initial_input_features.copy()

    #     for hour in range(24):
    #         setpoint_response = self.generate_optimized_setpoint(self.test_df.iloc[:48][["indoor_air_temperature", "cooling_set_point"]].values.tolist())
    #         if 'optimal_cooling_setpoint' in setpoint_response:
    #             input_features[0] = setpoint_response['optimal_cooling_setpoint']
    #             predicted_temp = self.predict_temperature(input_features)
    #             control_results.append({'hour': hour + 1, 'setpoint': input_features[0], 'predicted_temp': predicted_temp})

    #     return control_results
    
    def conduct_24h_control(self):
        """
        Conduct 24-hour HVAC control loop using the test dataset.
        The first 48 hours of test_df are used for initial state,
        and control is conducted for the next 24 hours.
        
        Returns:
            list: Control results for each hour
        """
        control_results = []
        
        # Initialize the 48-hour sliding window from first 48 hours of test data
        current_48h_series = self.test_df.iloc[:48][["indoor_air_temperature", "cooling_set_point"]].values.tolist()
        
        # Initialize current features from the first control timestep (hour 48)
        current_features = [
            self.test_df.iloc[48]['cooling_set_point'],      # cooling_set_point
            self.test_df.iloc[48]['temperature'],            # outdoor temperature
            self.test_df.iloc[48]['RH'],                     # relative humidity
            self.test_df.iloc[48]['hour'],                   # hour
            self.test_df.iloc[48]['day_of_week'],            # day_of_week
            self.test_df.iloc[47]['indoor_air_temperature'], # lag_1
            self.test_df.iloc[46]['indoor_air_temperature'], # lag_2
            self.test_df.iloc[45]['indoor_air_temperature']  # lag_3
        ]
        
        # Control loop for next 24 hours starting from hour 48
        for hour in range(24):
            current_hour_idx = 48 + hour  # Start from hour 48 in test_df
            
            # Generate optimized setpoint
            setpoint_response = self.generate_optimized_setpoint(current_48h_series)
            
            if 'optimal_cooling_setpoint' in setpoint_response:
                # Update features from test dataset
                current_features[1] = self.test_df.iloc[current_hour_idx]['temperature']
                current_features[2] = self.test_df.iloc[current_hour_idx]['RH']
                current_features[3] = self.test_df.iloc[current_hour_idx]['hour']
                current_features[4] = self.test_df.iloc[current_hour_idx]['day_of_week']
                
                # Update cooling setpoint from optimization
                current_features[0] = setpoint_response['optimal_cooling_setpoint']
                
                # Predict new indoor temperature
                predicted_temp = self.predict_temperature(current_features)
                
                # Update the sliding window
                current_48h_series.pop(0)  # Remove oldest hour
                current_48h_series.append([predicted_temp, setpoint_response['optimal_cooling_setpoint']])
                
                # Update temperature lags for next iteration
                current_features[-1] = current_features[-2]  # Update lag_3 = old lag_2
                current_features[-2] = current_features[-3]  # Update lag_2 = old lag_1
                current_features[-3] = predicted_temp        # Update lag_1 = latest prediction
                
                # Store results
                control_results.append({
                    'hour': hour + 1,
                    'timestamp': self.test_df.iloc[current_hour_idx]['timestamp'],
                    'outdoor_temp': current_features[1],
                    'RH': current_features[2],
                    'setpoint': setpoint_response['optimal_cooling_setpoint'],
                    'predicted_temp': predicted_temp,
                    'actual_temp': self.test_df.iloc[current_hour_idx]['indoor_air_temperature'],  # for comparison
                    'lags': {
                        'lag_1': current_features[-3],
                        'lag_2': current_features[-2],
                        'lag_3': current_features[-1]
                    }
                })
        
        return control_results


hvac_system = phiGPT('./vav_2-356.csv')

# test_df = hvac_system.test_df
# initial_input_features = test_df.iloc[47,1:-1].tolist() 
# sp = hvac_system.generate_optimized_setpoint(test_df.iloc[:48][["indoor_air_temperature", "cooling_set_point"]].values.tolist(), initial_input_features)

control_results = hvac_system.conduct_24h_control()

retest = test_df.iloc[49,1:-1].tolist()
a = hvac_system.predict_temperature(retest)


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


