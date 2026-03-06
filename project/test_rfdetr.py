import sys
try:
    from rfdetr import RFDETRBase
    model = RFDETRBase(pretrained=True, model_id='Subh775/Threat-Detection-RFDETR')
    import cv2, numpy as np
    img = np.zeros((640,640,3), dtype=np.uint8)
    res = model.predict(img)
    print("RES:")
    print(res)
    print(type(res))
    if hasattr(res, '__dict__'):
        print(res.__dict__)
except Exception as e:
    import traceback
    traceback.print_exc()
