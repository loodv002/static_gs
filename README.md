### Install Dependencies
```bash
$ pip install -r requirements.txt
```

### File Architecture
1. Place input ply files (`time_%05d.ply`) in `0448_ply`.
   
    ```
    0448_ply
     ├ time_00000.ply
     ├ time_00001.ply
     ...
    ```

2. Place static masks in `0448_masks_static`, organized by camera, as follow:

    ```
    0448_masks_static
     ├ 001001            <---- camera name
     |  ├ 00000.png      <---- frame id
     |  ├ 00001.png
     |  ...
     |
     ├ 002001
     |  ├ 00000.png
     |  ...
     |
     ...
    ```

3. If needed, replace `transforms.json`. Note that these entries are required: `transofrm_matrix`, `k1~4`, `fl_x`, `fl_y`, `cx`, `cy`, `h`, `w`. 

### Run
```bash
$ python process.py
```