image_cut.py 辨識批量照片，並進行部位切割

 line 45 在此目錄指定圖片集的目錄
 line 113-134 可修改儲存照片的名稱及路徑

./cut/part_cut/...
(0-5)類  上半身(upper)
(6-8)類  下半身(down)
(9-12)類 一件式(entire)

./cut/main_cut/
儲存裁切好的圖片

路徑 yolo/df2cfg/df2.names 保存類型名稱的檔案
路徑 yolo/utils/get_yolo_featvec.py 類型名稱對應的py檔，line 46-58 
	