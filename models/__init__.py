def cidnet():
    from .CIDNet import CIDNet
    model = CIDNet()
    return model

def retinexformer():
    from .RetinexFormer import RetinexFormer
    model = RetinexFormer()
    return model