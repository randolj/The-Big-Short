## Generating Model Files

1. From the root of the project (`The-Big-Short`), run the model generation script:

   ```bash
   python model.py
   ```

2. After the model files are generated, move them into the backend folder.

## Running the Web App

1. Navigate to the backend folder:

   ```bash
   cd big-short-web/backend
   ```

2. Start the backend server:

   ```bash
   uvicorn main:app --reload
   ```

3. Open a new terminal window and start the frontend:

   ```bash
   cd big-short-web
   npm start
   ```
