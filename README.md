# Melanoma Cancer Image Prediction - ETHGlobal Brussels

This project, developed by Luca Brugaletta and Enrique Barrueco at ETHGlobal Brussels, utilizes a neural network to predict whether skin melanomas are benign or malign based on high-resolution images. It's designed for individuals who wish to privately check their skin tumors before consulting a medical professional.

## Dataset

The model is trained on the [Melanoma Cancer Image Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset) which consists of 13,900 high-resolution images of both benign and malign tumors.

## Model

The neural network was developed using `nada_ai` from Nillion and was trained to meet size constraints while maintaining robust performance metrics:

- **Precision**: 88.24%
- **Recall**: 81.92%
- **F1 Score**: 84.96%
- **Accuracy**: 87.25%

The model, consisting of 52,744 weights, is stored securely as secrets on Nillion.

## Hosting

The model is hosted on the Nillion testnet, and we employ a Streamlit webapp allowing users to upload a skin tumor image and quickly receive a prediction. The service is currently free, with transaction costs on the testnet covered by our wallet.

## Usage

To use the prediction service:

1. Visit our Streamlit webapp.
2. Upload an image of the skin tumor.
3. Receive your prediction.

The image is processed privately without being stored on any server, ensuring user data remains confidential.

## Contributions

Contributions to this project are welcome. You can contribute in the following ways:

- Improving the model's accuracy and efficiency.
- Enhancing the web interface.
- Expanding the dataset with more diverse image samples.

For more details on how to contribute, please refer to the [contributing guidelines](CONTRIBUTING.md).

