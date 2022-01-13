
wget https://www.dropbox.com/s/r8o6pr8xus60vzz/max-acc.pth?dl=1

python3 test_testcase.py --load ./max-acc.pth?dl=1 --test_csv $1 --test_data_dir $2  --testcase_csv $3 --output_csv $4

exit 0











