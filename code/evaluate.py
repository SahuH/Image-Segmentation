model = unet(input_size=(256,256,1))
weight_path = './logs/unet_weights.best.hdf5'
model.load_weights(weight_path)
print(model)
dependencies = {
    'dice_coef': dice_coef

}
path = './logs/unet.h5' 
