from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from torchvision.transforms import functional as F
# from torchvision.transforms import  Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
from torch.utils.data import DataLoader
import torch
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, ColorJitter, RandomRotation
from PIL import Image
import numpy as np
from datasets import Dataset
import joblib
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os 

def random_rotation_torch(image, max_angle=15):
    """
    Apply a random rotation to an image using PyTorch functional transforms.

    Args:
    - image: Input image as a PIL.Image.
    - max_angle: Maximum rotation angle in degrees.

    Returns:
    - Rotated image.
    """
    angle = random.uniform(-max_angle, max_angle)
    return F.rotate(image, angle)



def vit_pretrained():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels = 2,
        id2label={i: f"class_{i}" for i in range(2)},  # Example class labels
        label2id={f"class_{i}": i for i in range(2)},
        ignore_mismatched_sizes = True
    )
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    feature_extractor.save_pretrained("/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/model/feature_extractor1")
    return model, feature_extractor

def resize_pic(image, target_size = (224, 224)):
    resized_image = []
    for img in image:
        #reshape the flatten image back to 2D
        img_2d = img.reshape(128,128)
        # Ensure the array is of type uint8 (if pixel values are 0-255)
        img_2d = img_2d.astype(np.uint8)
        img_resized = Image.fromarray(img_2d, mode='L')
        # Convert to RGB and resize to 224x224
        img_resized = img_resized.convert("RGB")
        img_resized = img_resized.resize(target_size)
        resized_image.append(np.array(img_resized))

    return np.array(resized_image)

def model_arguments_trainer(X_train_normalized, X_test_normalized, y_train_tensor, y_test_tensor):

    train_dataset = Dataset.from_dict({
        'pixel_values': X_train_normalized.numpy(),
        'labels': y_train_tensor.numpy()
    })

    eval_dataset = Dataset.from_dict({
        'pixel_values': X_test_normalized.numpy(),
        'labels': y_test_tensor.numpy()
    })

    model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(np.unique(y_test_tensor)),  # Number of classes
    ignore_mismatched_sizes = True
)
    # Define training arguments
    training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs"
    )   
    # Trainer setup
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=None
    )
    # Train the model
    trainer.train()
    # Save the trained model
    model.save_pretrained("/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/model/vit-image-classification1")
    # Save the trainer's state
    trainer.save_state()

    # Save the trainer's arguments
    # trainer.args.save_to_json("/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/model/training_args.json")

    return model, trainer


def normalization(image, feature_extractor):
    inputs = feature_extractor(image, return_tensors = "pt")
    return inputs['pixel_values']

def prepare_df(X_train, X_test, y_train, y_test, feature_extractor):
    X_train_resized = resize_pic(X_train, target_size=(224, 224))
    X_test_resized = resize_pic(X_test, target_size=(224, 224))

    X_train_normalized = normalization(X_train_resized, feature_extractor)
    X_test_normalized = normalization(X_test_resized, feature_extractor)

    y_train_tensor = torch.tensor(y_train, dtype = torch.long)
    y_test_tensor = torch.tensor(y_test, dtype = torch.long)
    return X_train_normalized, X_test_normalized, y_train_tensor, y_test_tensor

def prepare_df_with_augmentation(X_train, X_test, y_train, y_test, feature_extractor):
    # Training data augmentation
    augmentation_transform = Compose([
        Resize((224, 224)),                   # Resize to ViT input size
        RandomHorizontalFlip(p=0.5),         # Random horizontal flip
        RandomRotation(degrees=15),          # Random rotation within ±15 degrees
        ColorJitter(brightness=0.2),         # Adjust brightness by ±20%
        ToTensor(),                          # Convert to PyTorch tensor
        Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)  # Normalize
    ])

    # Test data minimal preprocessing
    test_transform = Compose([
        Resize((224, 224)),                   # Resize to ViT input size
        ToTensor(),                          # Convert to PyTorch tensor
        Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)  # Normalize
    ])

    # Convert and transform training data
    X_train_augmented = torch.stack([
        augmentation_transform(Image.fromarray(img).convert("RGB")) for img in X_train
    ])
    # Convert and transform test data
    X_test_transformed = torch.stack([
        test_transform(Image.fromarray(img).convert("RGB")) for img in X_test
    ])

    # Convert labels to tensors
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_augmented, X_test_transformed, y_train_tensor, y_test_tensor

def predict_evaluate(model, feature_extractor, X_test_normalized, y_test_tensor,
                    return_probabilities=False, return_predictions=False):
    model.eval()
    #decide hardware to run
    device = (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    model.to(device)
    X_test_normalized = X_test_normalized.to(device)

    #make prediction
    with torch.no_grad():
        outputs = model(X_test_normalized)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        prediction = torch.argmax(outputs.logits, dim = 1).cpu().numpy()
    
    if return_probabilities:
        return probs[:, 1]
    elif return_predictions:
        return prediction
    else: 
        return None

def compute_f1_score(y_test_tensor, prediction):
        #calculated weighted F1 Score
        weighted_f1_score = f1_score(y_test_tensor, prediction, average = 'weighted')
        print("Weighted F1-Score:", weighted_f1_score)
        return weighted_f1_score

def ROC_Curve(y_test_tensor, probabilities):
    plt.figure(figsize=(8, 6))
    fpr, tpr, threshold = roc_curve(y_test_tensor, probabilities)
    auc_k = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0,1], [0,1], 'k--')
    plt.plot(fpr, tpr, label = ' (area = {:.3f})'.format(auc_k))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

def plot_confusion_matrix(y_test_tensor, prediction):
    cm = confusion_matrix(y_test_tensor, prediction)
    disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
    disp.plot(cmap='viridis')
    plt.title('Confusion Matrix')
    plt.show()

    

if __name__ == '__main__':
    file_path = '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/split_data_wo_gan.pkl'
    #load model and feature extractor
    model, feature_extractor = vit_pretrained()
     # Resize and apply augmentation
    X_train, X_test, y_train, y_test = joblib.load(file_path)
    X_train_augmented, X_test_transformed, y_train_tensor, y_test_tensor = prepare_df_with_augmentation(
        X_train, X_test, y_train, y_test, feature_extractor
    )
    print(f"Augmented X_train: {X_train_augmented.shape}")
    print(f"Transformed X_test: {X_test_transformed.shape}")

    # Convert tensor back to image
    image = X_train_augmented[0].permute(1, 2, 0).numpy()  # Change dimension order for display
    image = (image * feature_extractor.image_std + feature_extractor.image_mean)  # De-normalize
    image = np.clip(image, 0, 1)  # Clip to valid range [0, 1]

    plt.imshow(image)
    plt.show()

    #train the dataset
    model, trainer = model_arguments_trainer(X_train_augmented, X_test_transformed, y_train_tensor, y_test_tensor)
    prediction = predict_evaluate(model, feature_extractor, X_test_transformed, y_test_tensor, return_probabilities=False, return_predictions=True)
    weighted_f1_score = compute_f1_score(y_test_tensor, prediction)
    print("Weighted F1-Score:", weighted_f1_score)
    plot_confusion_matrix(y_test_tensor.cpu().numpy(), prediction)
    probabilities = predict_evaluate(model, feature_extractor, X_test_transformed, y_test_tensor, return_probabilities=True, return_predictions = False)
    ROC_Curve(y_test_tensor.cpu().numpy(), probabilities)




    # # Resize train and test images
    # X_train, X_test, y_train, y_test = joblib.load(file_path)
    # X_train_normalized, X_test_normalized, y_train_tensor, y_test_tensor = prepare_df(X_train, X_test, y_train, y_test, feature_extractor)

    # print(f"Resized X_train: {X_train_normalized.shape}")
    # print(f"Resized X_test: {X_test_normalized.shape}")

    # #train the dataset
    # model, trainer = model_arguments_trainer(X_train_normalized, X_test_normalized, y_train_tensor, y_test_tensor)
    # prediction = predict_evaluate(model, feature_extractor, X_test_normalized, y_test_tensor, return_probabilities=False, return_predictions=True)
    # weighted_f1_score = compute_f1_score(y_test_tensor, prediction)
    # print("Weighted F1-Score:", weighted_f1_score)
    # plot_confusion_matrix(y_test_tensor.cpu().numpy(), prediction)
    # probabilities = predict_evaluate(model, feature_extractor, X_test_normalized, y_test_tensor, return_probabilities=True, return_predictions = False)
    # ROC_Curve(y_test_tensor.cpu().numpy(), probabilities)
