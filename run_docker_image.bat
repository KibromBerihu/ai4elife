echo '[1] Running the whole pipeline in testing mode...'

echo '[2] Kindly make sure that you meet the folder structure arrangement given in readme.md ...'

echo '[3] Path to the raw pet and ground truth (gt)  folders, e.ge. E:/data/ Where the data
        is the main path directory to all cases'
set input_dir=%1

echo '[4] Path to the output folders, e.ge. E:/output/ where the all results should be saved'
set output_dir=%2

echo '[5] Unique docker image name (Name used during building):, e.g., ai4elife:v1, put ai4elife '
set docker_image_name=%3

echo '[6] Unique docker tage (Name used during building) otherwise latest will be downloaded:, e.g., ai4elife:v1, give v1'
set docker_tag=%4

echo '[7] Unique container id (any name): '
set CONTAINERID=%5

echo '[8] Running the docker with container id: %CONTAINERID% ....'

docker run -it --rm --name %CONTAINERID%^
    -v %input_dir%:/input ^
    -v %output_dir%:/output ^
    %docker_image_name%:%docker_tag% ^

