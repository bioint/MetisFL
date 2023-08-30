
from metisfl.driver.driver_session import DriverSession
from env import env

session = DriverSession(env)

res = session.run()

print(res)
