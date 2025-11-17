# spark/preprocess.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg
import os
from dotenv import load_dotenv # Import dotenv to load .env file

# Load environment variables from .env file
load_dotenv()

# Get the path to the data file directly
#DATA_PATH = os.getenv("DATA_PATH", "data/student_sample.csv")
# Replace the line where DATA_PATH is defined with:
DATA_PATH = "/opt/spark/data/student_sample.csv"
def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("StudentRetentionPreprocessing") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading data from: {DATA_PATH}")
    # Load the data
    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

    print("--- Initial Data Info ---")
    print(f"Total Records: {df.count()}")
    print("\nSchema:")
    df.printSchema()
    print("\nSample Data:")
    df.show(5, truncate=False)

    print("\n--- Data Quality Check ---")
    # Check for missing values in key columns
    for col_name in df.columns:
        null_count = df.filter(col(col_name).isNull()).count()
        if null_count > 0:
            print(f"Column '{col_name}' has {null_count} null values.")

    print("\n--- Feature Engineering ---")
    # Example: Calculate approval rate for 1st semester
    # Use the exact column name from the schema: "Curricular units 1st sem (enrolled)"
    df = df.withColumn(
        "approval_rate_1st_sem",
        when(col("Curricular units 1st sem (enrolled)") > 0, # Fixed column name
             col("Curricular units 1st sem (approved)") / col("Curricular units 1st sem (enrolled)")) # Fixed column names
        .otherwise(0.0) # Handle division by zero
    )

    # Example: Create a feature for academic performance (average of grades if available)
    # Note: You have grades for 1st and 2nd sem. You could average them or use just 1st.
    df = df.withColumn(
        "avg_grade",
        (col("Curricular units 1st sem (grade)") + col("Curricular units 2nd sem (grade)")) / 2.0 # Fixed column names
    ).fillna({"avg_grade": 0.0}) # Fill NaN if both grades are missing

    # Example: Flag for financial risk (debtor AND not up-to-date on fees)
    df = df.withColumn(
        "financial_risk_flag",
        when((col("Debtor") == 1) & (col("Tuition fees up to date") == 0), 1).otherwise(0) # Fixed column names
    )

    print("\nSchema After Feature Engineering:")
    df.printSchema()
    print("\nSample Data After Feature Engineering:")
    # Select using the exact original column names and the new feature names
    df.select(
        "Marital status", "Course", "Target", "approval_rate_1st_sem", 
        "avg_grade", "financial_risk_flag"
    ).show(10, truncate=False)

    # Optional: Save the processed DataFrame to a Parquet file for later use
    output_path = "data/processed_student_data.parquet"
    print(f"\n--- Saving Processed Data to {output_path} ---")
    df.write.mode("overwrite").parquet(output_path)
    print("Data saved successfully!")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()