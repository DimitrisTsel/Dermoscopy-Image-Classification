### Libraries
import glob

### Data I/O
def findScan(data, name, key):
    """"
    Input: 
        data     - dict of dict to sort the data
        key      - key of 'data' ('id','image','label')
        value    - value of 'key'
    Output:
        value
    """
    for i, dic in data.items():
        if dic[key] == name:        
            return i
    return -1



def sortData(path, mode='train-val', mask=False, mask_mode=False):
    """"
    Input:  path
    Output: data[p] = {
        'id':
        'image':
        'label': }
        
    """
    if (mode=='train-val'):

        # Importing Images
        les_dir = glob.glob(path+"/malignant/*.jpg")
        nv_dir  = glob.glob(path+"/benign/*.jpg")
        print("Number of LES Images:", len(les_dir))
        print("Number of NV Images:",  len(nv_dir))

        
        if (mask==False):
            # Creating Dictionary (LES Scans)
            data = {}
            for p in range(len(les_dir)):
                scan_id  =  les_dir[p].replace(".jpg", "")
                scan_id  =  scan_id.replace(path+"/les\\", "")

                # Creating List of Dictionary                    
                data[p] = {
                            'id'    : scan_id,
                            'image' : les_dir[p],
                            'label' : 1 }            # Label of LES = 1

            # Creating Dictionary (NV Scans)
            for p in range(len(nv_dir)):
                scan_id  =  nv_dir[p].replace(".jpg", "")
                scan_id  =  scan_id.replace(path+"/nv\\", "")

                # Creating List of Dictionary                    
                data[p+len(les_dir)] = {
                            'id'    : scan_id,
                            'image' : nv_dir[p],
                            'label' : 0 }            # Label of NV = 0

    if (mode=='test'):
        
        # Importing Images
        target_dir  = glob.glob(path+"/*.jpg")
        print("Number of Test Images:", len(target_dir))
        
        # Creating Dictionary
        data = {}
        for p in range(len(target_dir)):
            scan_id  =  target_dir[p].replace(".jpg", "")
            scan_id  =  scan_id.replace(path+"\\", "")

            # Creating List of Dictionary                    
            data[p] = {
                        'id'    : scan_id,
                        'image' : target_dir[p] }
    return data


