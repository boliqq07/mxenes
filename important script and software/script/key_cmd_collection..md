
复制vasp必要文件
-----------------

    # 解压，复制
    tar -zcvf data.tar.gz ./ --exclude=AECC* --exclude=CHG* --exclude=WAVE* --exclude=PROCAR --exclude=*err* --exclude=*out*
    # 解压
    tar -zcvf data.tar.gz
