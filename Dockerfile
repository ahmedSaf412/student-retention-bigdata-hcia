# Use the official Apache Spark image as base
FROM apache/spark:3.5.3

# Switch to root to install packages
USER root

# Install Python dependencies needed for your project
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install python-dotenv scikit-learn pandas joblib numpy imbalanced-learn

# Switch back to the spark user
USER spark