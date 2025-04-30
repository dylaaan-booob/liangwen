from src.utils import *

now_time = time.strftime('%m-%d %H-%M', time.localtime())
zip_files_only(get_dir_name(True) + "/need_zip_dir", get_dir_name(True) + "/history_zip_file/" + now_time + ".zip")
