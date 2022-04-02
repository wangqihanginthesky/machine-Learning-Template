# machine-Learning-Template

### About This Repository
- This repository maintains some of my own commonly used machine learning related code.
- This repository only provide a baseline. You need to custimize your own model.
- This repository also support you to create a web server by FastApi.

### How to use
- step1 build image 
    ```
    docker build -t kaggle_template .
    ```
- step2 start the container & jupyter notebook
    ```
    #Mount to your local env
    docker run --rm --name notebook -it -p 8888:8888 -p 4000:5000 -v ${PWD}/table:/opt kaggle_template /bin/bash
    jupyter-lab --ip 0.0.0.0 --allow-root
    ```

- step3 start mlflow server
    ```
    docker exec -it notebook /bin/bash
    mlflow ui --host=0.0.0.0
    ```
