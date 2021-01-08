import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)

planeStartPos = [0, 0, 0]
planeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
planeId = p.loadURDF("plane.urdf", planeStartPos, planeStartOrientation)


gripperStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
gripperStartPos = [0, 0, 0]
#gripperId1 = p.loadURDF("robotiq_2f_85.urdf", gripperStartPos, gripperStartOrientation)
gripperId1 = p.loadURDF("robotiq_2f_85_orig.urdf", gripperStartPos, gripperStartOrientation)

# Set free joints using pybullet.setJointMotorControl2

try:
    while True:
        p.stepSimulation()
    	time.sleep(1./100.)
except KeyboardInterrupt:
    pass


print(" ")
print("Joint Info:")
print(p.getJointInfo(gripperId1, 0) )
print(" ")

print("Dynamics Info:")
print(p.getDynamicsInfo(gripperId1, 0) )
print(" ")

p.disconnect()

