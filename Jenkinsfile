pipeline {
    agent any

    environment {
        PORT = "8009"
        DOCKER_IMAGE = "cuda-fastapi-app:latest"
    }

    stages {

        stage('GPU Sanity Test') {
            steps {
                echo 'Installing required Python dependencies...'
                sh '''
                    python3 -m pip install --upgrade pip
                    pip install fastapi uvicorn numpy numba
                '''

                echo 'Running CUDA sanity check...'
                sh '''
                    python3 -c "import numba; from numba import cuda; print('CUDA detected:', cuda.is_available())"
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image with GPU support..."
                sh '''
                    docker build -t $DOCKER_IMAGE .
                '''
            }
        }

        stage('Deploy Container') {
            steps {
                echo "Deploying Docker container..."
                sh '''
                    # Stop previous container if exists
                    docker rm -f cuda-fastapi-container || true

                    # Run new container with GPU access
                    docker run --gpus all -d --name cuda-fastapi-container -p $PORT:$PORT $DOCKER_IMAGE
                '''
            }
        }
    }

    post {
        success {
            echo "Deployment completed successfully!"
        }
        failure {
            echo " Deployment failed. Check logs for errors."
        }
        always {
            echo "Pipeline finished."
        }
    }
}
