import tensorflow as tf

# Replace this with the correct absolute path to your saved model
model_path = "model"

# Load the saved model from the .pb file
loaded_model = tf.saved_model.load(model_path)

# Print the input and output signatures of the model
print("Input Signature:", loaded_model.signatures["serving_default"].inputs)
print("Output Signature:", loaded_model.signatures["serving_default"].outputs)
