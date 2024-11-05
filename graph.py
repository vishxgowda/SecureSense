import matplotlib.pyplot as plt

# Example data (Replace this with your actual training metrics)
# You should replace these lists with the actual values collected during training
train_loss = [0.8, 0.6, 0.4, 0.3, 0.2]  # Example training loss over epochs
val_loss = [0.9, 0.7, 0.5, 0.4, 0.3]    # Example validation loss over epochs
train_accuracy = [0.5, 0.6, 0.7, 0.8, 0.9]  # Example training accuracy over epochs
val_accuracy = [0.4, 0.5, 0.6, 0.7, 0.8]    # Example validation accuracy over epochs

# Generate the epochs based on the length of the training metrics
epochs = range(1, len(train_loss) + 1)

# Plotting Training and Validation Loss
plt.figure(figsize=(12, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
