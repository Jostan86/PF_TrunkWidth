import rospy
from std_msgs.msg import Int32
from concurrent.futures import ThreadPoolExecutor

class Trial4:
    def __init__(self):
        self.num_sub = rospy.Subscriber("/num_pub", Int32, self.num_callback)
        self.nums = []
        self.executor = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers as needed

    def num_callback(self, num_msg):
        self.nums.append(num_msg.data)
        num_msgs = len(self.nums)
        if num_msgs < 10:
            print(num_msgs)
            self.executor.submit(self.trial_func, num_msgs)

    def trial_func(self, num):
        try:
            print("Starting trial_func: ", num)
            for i in range(100000000):  # Adjust the range or computation logic as needed
                pass
            print("Done with computation: ", num)
        except Exception as e:
            print(f"Exception in trial_func: {e}")

if __name__ == "__main__":
    rospy.init_node("trial4")
    Trial4()
    rospy.spin()



