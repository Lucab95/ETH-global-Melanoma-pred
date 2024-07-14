# Melanoma Cancer Image Prediction - ETHGlobal Brussels

This project, developed by Luca Brugaletta and Enrique Barrueco at ETHGlobal Brussels, utilizes a neural network to predict whether skin melanomas are benign or malign based on high-resolution images. It's designed for individuals who wish to privately check their skin tumors before consulting a medical professional.

## Dataset

The model is trained on the [Melanoma Cancer Image Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset) which consists of 13,900 high-resolution images of both benign and malign tumors.

## Model

The neural network was developed using `nada_ai` from Nillion and was trained to meet size constraints while maintaining robust performance metrics:

- **Precision**: 80.607%
- **Recall**: 76.327%
- **F1 Score**: 78.409%
- **Accuracy**: 76.625%

The model, consisting of 94 weights, is stored securely as secrets on Nillion.

In order to host the model on Nillion one must run the provider.ipynb notebook.

## Hosting

The model is hosted on the Nillion testnet, and we employ a Streamlit webapp allowing users to upload a skin tumor image and quickly receive a prediction. The service is currently free, with transaction costs on the testnet covered by our wallet.

## Usage

To use the prediction service, please follow the isntruction on the next step:

### Inference 

1. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate 
```

2. Install all the required library by:
```
 pip install -r requirements.txt
 ```
3. Create an .env file according to `https://docs.nillion.com/network-configuration` and place it inside `/nillion/quickstart/nada_quickstart_programs`
4. Run the streamlit platfor with:
 ```
 Streamlit run steam.py
 ```
5. Upload an image of a skin area.
6. Receive your prediction.

The image is processed privately without being stored on any server, ensuring user data remains confidential.

## Contributions

Contributions to this project are welcome. You can contribute in the following ways:

- Improving the model's accuracy and efficiency.
- Enhancing the web interface.
- Expanding the dataset with more diverse image samples.

