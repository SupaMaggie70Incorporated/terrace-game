In order to profile, you must run the following command
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
This is bad for security, so once done, reset using the following command
echo 4 | sudo tee /proc/sys/kernel/perf_event_paranoid