import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import os

class UAVDataPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def ingest_data(self):
        """Loads data with error handling."""
        try:
            print("Loading dataset...")
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Initial shape: {self.df.shape}")
        except FileNotFoundError:
            print(f"Error: Could not find file at {self.data_path}. Please ensure it is in the correct directory.")
        except Exception as e:
            print(f"An unexpected error occurred during ingestion: {e}")

    def clean_data(self):
        """Detects missing values, removes duplicates, and corrects data types."""
        print("Cleaning data...")
        if self.df is not None:
            # 1. Remove duplicate records
            initial_len = len(self.df)
            self.df.drop_duplicates(inplace=True)
            print(f"Dropped {initial_len - len(self.df)} duplicate rows.")
            
            # 2. Handle missing/null values
            missing_count = self.df.isnull().sum().sum()
            if missing_count > 0:
                self.df.dropna(inplace=True)
                print(f"Dropped rows with missing values. Total missing values found: {missing_count}")
            else:
                print("No missing values found.")
                
            # 3. Correct data types (ensure labels is int, others are float)
            self.df['labels'] = self.df['labels'].astype(int)
            print("Data types validated.")

    def apply_unique_filter(self):
        """Applies a unique programmatic filter to ensure a unique data slice."""
        print("Applying unique filter logic...")
        if self.df is not None:
            # UNIQUE FILTER LOGIC:
            # 1. Keep only Label 0 (Normal) and Label 1 (GPS Anomaly)
            self.df = self.df[self.df['labels'].isin([0, 1])]
            
            # 2. To ensure mathematical uniqueness, filter out the initial start-up phase 
            # (e.g. drop the first 1000 rows where timestamp might represent grounded calibration)
            self.df = self.df.sort_values(by='timestamp').iloc[1000:]
            
            # Save the cleaned dataset
            os.makedirs("data", exist_ok=True)
            self.df.to_csv("data/dataset_cleaned.csv", index=False)
            print(f"Unique filter applied. Final dataset shape: {self.df.shape}. Saved to dataset_cleaned.csv")

    def analyze_data(self):
        """Computes engineering data analytics metrics using NumPy."""
        print("\n--- ENGINEERING DATA ANALYTICS ---")
        if self.df is not None:
            # 1. Descriptive Statistics using NumPy on Roll and Pitch
            roll_data = self.df['Roll'].to_numpy()
            pitch_data = self.df['Pitch'].to_numpy()
            
            print(f"Roll - Mean: {np.mean(roll_data):.4f}, Median: {np.median(roll_data):.4f}")
            print(f"Roll - Std Dev: {np.std(roll_data):.4f}, Variance: {np.var(roll_data):.4f}")
            print(f"Pitch - Mean: {np.mean(pitch_data):.4f}, Median: {np.median(pitch_data):.4f}")
            print(f"Pitch - Std Dev: {np.std(pitch_data):.4f}, Variance: {np.var(pitch_data):.4f}")
            
            # 2. Comparative Analysis (Normal vs GPS Anomaly)
            normal_data = self.df[self.df['labels'] == 0]['ErrRP'].to_numpy()
            anomaly_data = self.df[self.df['labels'] == 1]['ErrRP'].to_numpy()
            
            if len(normal_data) > 0 and len(anomaly_data) > 0:
                print(f"\nComparative Analysis (Error in Roll/Pitch):")
                print(f"Normal Flight Mean Error: {np.mean(normal_data):.4f}")
                print(f"GPS Anomaly Mean Error: {np.mean(anomaly_data):.4f}")
            
            # 3. Correlation Analysis
            corr_matrix = self.df[['Roll', 'Pitch', 'Yaw', 'ErrRP', 'labels']].corr()
            print("\nCorrelation Matrix (Selected Variables):")
            print(corr_matrix)
            print("----------------------------------\n")

    def visualize_data(self):
        """Generates static and animated visualizations."""
        print("Generating Visualizations...")
        if self.df is not None:
            os.makedirs("outputs", exist_ok=True)
            
            # 1. Static Graph: Correlation Heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(self.df.corr(), cmap='coolwarm', annot=False)
            plt.title('Sensor Telemetry Correlation Heatmap')
            plt.tight_layout()
            plt.savefig("outputs/heatmap.png")
            plt.close()
            
            # 2. Static Graph: Boxplot for Roll/Pitch Errors by Label
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='labels', y='ErrRP', data=self.df)
            plt.title('Roll/Pitch Error Distribution (0=Normal, 1=GPS Anomaly)')
            plt.savefig("outputs/boxplot_error.png")
            plt.close()
            
            # 3. Static Graph: Scatter plot of Desired vs Actual Roll
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x='DesRoll', y='Roll', hue='labels', data=self.df, alpha=0.5)
            plt.title('Desired Roll vs Actual Roll')
            plt.savefig("outputs/scatter_roll.png")
            plt.close()
            print("Static graphs saved to outputs/ folder.")
            
            # Downsample for animations to prevent performance issues
            df_anim = self.df.iloc[::20].copy() # take every 20th row
            df_anim['seq'] = range(len(df_anim))
            
            # 4. Animated Graph: Time Series of Roll over time
            fig1 = px.scatter(df_anim, x="timestamp", y="Roll", color="labels", 
                              animation_frame="seq", 
                              range_x=[df_anim['timestamp'].min(), df_anim['timestamp'].max()],
                              range_y=[df_anim['Roll'].min(), df_anim['Roll'].max()],
                              title="Animated Roll Dynamics Over Time")
            fig1.write_html("outputs/animated_roll_timeseries.html")
            
            # 5. Animated Graph: 3D Scatter of MagX, MagY, MagZ shifting
            fig2 = px.scatter_3d(df_anim, x='MagX', y='MagY', z='MagZ', color='labels',
                                 animation_frame='seq', range_x=[df_anim['MagX'].min(), df_anim['MagX'].max()],
                                 range_y=[df_anim['MagY'].min(), df_anim['MagY'].max()],
                                 range_z=[df_anim['MagZ'].min(), df_anim['MagZ'].max()],
                                 title="Animated Magnetometer Drift 3D")
            fig2.write_html("outputs/animated_mag_drift.html")
            print("Animated graphs saved as HTML files in outputs/ folder.")

if __name__ == "__main__":
    pipeline = UAVDataPipeline("data/dataset_original.csv")
    pipeline.ingest_data()
    pipeline.clean_data()
    pipeline.apply_unique_filter()
    pipeline.analyze_data()
    pipeline.visualize_data()
    print("Pipeline Execution Completed Successfully.")
