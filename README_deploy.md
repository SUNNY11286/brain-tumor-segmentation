# Brain Tumor Segmentation Flask App

This repository contains a Flask application that uses a trained model to predict brain tumor segmentation masks and display the results on a web page.

## Features

- Upload brain MRI images
- Predict brain tumor segmentation masks using a trained model
- Display the original image and the predicted mask on a web page

## Requirements

- Python 3.8+
- Flask
- TensorFlow/Keras
- NumPy
- OpenCV
- Heroku CLI

## Setup

### Step 1: Prepare Your Flask App for Deployment

1. **Create a `requirements.txt` file** to list all dependencies:
    ```bash
    pip freeze > requirements.txt
    ```

2. **Create a `Procfile`** to specify the commands that are executed by the app on startup:
    ```Procfile
    web: python app.py
    ```

3. **Update `app.py`** to use the correct host and port:
    ```python:app.py
    if __name__ == '__main__':
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    ```

4. **Create a `runtime.txt`** to specify the Python version:
    ```runtime.txt
    python-3.8.10
    ```

### Step 2: Push Your Code to GitHub

1. **Initialize a Git repository** (if you haven't already):
    ```bash
    git init
    git add .
    git commit -m "Initial commit"
    ```

2. **Create a new repository on GitHub** and follow the instructions to push your local repository to GitHub:
    ```bash
    git remote add origin https://github.com/yourusername/your-repo-name.git
    git branch -M main
    git push -u origin main
    ```

### Step 3: Deploy to Heroku

1. **Install the Heroku CLI** if you haven't already. Follow the instructions on the [Heroku CLI installation page](https://devcenter.heroku.com/articles/heroku-cli).

2. **Login to Heroku**:
    ```bash
    heroku login
    ```

3. **Create a new Heroku app**:
    ```bash
    heroku create your-app-name
    ```

4. **Deploy your code to Heroku**:
    ```bash
    git push heroku main
    ```

5. **Open your app in the browser**:
    ```bash
    heroku open
    ```

### Step 4: Link Heroku with GitHub for Continuous Deployment

1. **Go to the Heroku Dashboard**, select your app, and navigate to the "Deploy" tab.

2. **Connect to GitHub**:
    - Under "Deployment method", select "GitHub".
    - Connect your GitHub account and select the repository you want to deploy.

3. **Enable Automatic Deploys** (optional):
    - This will automatically deploy your app whenever you push to the main branch on GitHub.

### Step 5: Update Your HTML Templates to Use GitHub Pages for Static Content (Optional)

If you want to serve static content (like HTML, CSS, and JavaScript) from GitHub Pages, you can:

1. **Create a `gh-pages` branch** in your repository:
    ```bash
    git checkout --orphan gh-pages
    git reset --hard
    ```

2. **Move your static files to this branch** and push:
    ```bash
    git add .
    git commit -m "Deploy static content"
    git push origin gh-pages
    ```

3. **Enable GitHub Pages** in the repository settings, and set the source to the `gh-pages` branch.

## Running the App Locally

To run the Flask application locally, use the following command:

```
python app.py
```


You can then access the application in your web browser at `http://127.0.0.1:5000/`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.