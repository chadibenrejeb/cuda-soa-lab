pipeline {
    agent any

    environment {
        PORT = "8009"
        DOCKER_IMAGE = "cuda-fastapi-app:latest"
        CONTAINER_NAME = "cuda-fastapi-container"
    }

    stages {

        stage('GPU Sanity Test') {
            steps {
                echo 'Running CUDA sanity check inside Docker...'
                sh '''
                    docker run --rm --gpus all $DOCKER_IMAGE \
                    python3 -c "from numba import cuda; print('CUDA detected:', cuda.is_available())"
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "🐳 Building Docker image with GPU support..."
                sh '''
                    docker build -t $DOCKER_IMAGE .
                '''
            }
        }

        stage('Deploy Container') {
            steps {
                echo "🚀 Deploying Docker container..."
                sh '''
                    # Stop previous container if exists
                    docker rm -f $CONTAINER_NAME || true

                    # Run new container with GPU access
                    docker run --gpus all -d --name $CONTAINER_NAME -p $PORT:$PORT $DOCKER_IMAGE
                '''
            }
        }

        stage('Test API Endpoints') {
            steps {
                echo "🧪 Testing FastAPI endpoints inside Docker container..."
                sh '''
                    # Health check
                    curl -s http://localhost:$PORT/health

                    # Test /add endpoint using your existing .npz files
                    curl -s -X POST -F "file_a=@matrix1.npz" -F "file_b=@matrix2.npz" http://localhost:$PORT/add
                '''
            }
        }
    }

    post {
        success {
            echo "🎉 Pipeline completed successfully!"
        }
        failure {
            echo "💥 Pipeline failed. Check logs for errors."
        }
        always {
            echo "🧾 Pipeline finished."
        }
    }
}
