import pickle
import timeit

# This is my own custom function for convenience in unpickling the preprocessed data.
def unpickle(pickleFilename:str, fromCreatedPicklesDirectory = True, loadAsDict=False):
    if loadAsDict is False:
        resultingList = []
    else:
        resultingList = dict()
    print(f"Start unpickle-ing the {pickleFilename} file")
    start_time = timeit.default_timer()
    if fromCreatedPicklesDirectory is True:
        with open (f'./createdPickles/{pickleFilename}', 'rb') as fp:
            resultingList = pickle.load(fp)
    else:
        with open (f'{pickleFilename}', 'rb') as fp:
            resultingList = pickle.load(fp)
    print(f"Finished unpickle-ing the {pickleFilename} file to the list, \n Time taken: ", timeit.default_timer() - start_time)
    return resultingList

# This is my own custom function for convenience in pickling the preprocessed data.
def pickleMyProgress(dataToSave, pickleFilename:str, saveToMyPicklesDirectory = True):
    print(f"Start pickle-ing to {pickleFilename} file")
    start_time = timeit.default_timer()
    if saveToMyPicklesDirectory is True:
        with open(f'./createdPickles/{pickleFilename}', 'wb') as fp:
            pickle.dump(dataToSave, fp)
    else:
        with open(f'{pickleFilename}', 'wb') as fp:
            pickle.dump(dataToSave, fp)
    print(f"Finished pickle-ing the {pickleFilename} \n Time taken: { timeit.default_timer() - start_time}")