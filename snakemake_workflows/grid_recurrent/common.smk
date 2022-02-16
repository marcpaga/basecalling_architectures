def get_model_name_folder(cnn, enc, dec, at, ap, m, win):

    m = flag_to_str(flag=m)
    model_name_folder = [
        cnn,
        enc,
        dec,
        at,
        ap,
        m,
        win
    ]

    model_name_folder = "_".join(model_name_folder)

    return model_name_folder

def flag_to_str(flag):

    if flag == '':
        return 'False'
    else:
        return 'True'

    

def bool_to_flag(flag, name):

    if flag == 'True':
        return name
    elif flag == 'False':
        return ''