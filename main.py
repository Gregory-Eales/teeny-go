from betago.processor import SevenPlaneProcessor
processor = SevenPlaneProcessor()
input_channels = processor.num_planes

# Load go data and one-hot encode labels
X, y = processor.load_go_data(num_samples=1000)
X = X.astype('float32')
Y = np_utils.to_categorical(y, nb_classes)
