import sys
import os.path

## Set path so that you can import the local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from queue import Queue
import threading
import time

LEARNERS = 10


# Define script's static namedtuple collections
class ExecutionInterruptionSignals(object):

	def __init__(self, target_learners=None, target_epochs=None, target_accuracy=None, target_exec_time_mins=None):
		self.target_learners = target_learners
		self.target_epochs = target_epochs
		self.target_accuracy = target_accuracy
		self.target_exec_time_mins = target_exec_time_mins

	def is_target_learners_reached(self, current_learners=None):
		if self.target_learners is not None and current_learners is not None:
			return current_learners >= self.target_learners

	def is_target_epochs_reached(self, current_epochs=None):
		if self.target_epochs is not None and current_epochs is not None:
			return current_epochs >= self.target_epochs

	def is_accuracy_reached(self, acc_val=None):
		if self.target_accuracy is not None and acc_val is not None:
			return acc_val >= self.target_accuracy

	def is_execution_time_reached(self, current_exec_time_mins=None):
		if self.target_exec_time_mins is not None and current_exec_time_mins is not None:
			return current_exec_time_mins >= self.target_exec_time_mins


class AsynchronousExecutionSignals(ExecutionInterruptionSignals):

	def __init__(self, target_learners=None, target_epochs=None, target_accuracy=None, target_exec_time_mins=None):
		ExecutionInterruptionSignals.__init__(self, target_learners, target_epochs, target_accuracy, target_exec_time_mins)
		self.__is_target_reached = False

	def set_target_signal(self):
		self.__is_target_reached = True

	def reset_target_signal(self):
		self.__is_target_reached = False

	def is_target_signal_reached(self):
		return self.__is_target_reached


class FederationExecutionRes(object):

	def __init__(self, completed_epochs=0, completed_batches=0, data_size=100, latest_weights=list()):
		self.completed_epochs = completed_epochs
		self.completed_batches = completed_batches
		self.data_size = data_size
		self.latest_weights = latest_weights


class qObject(object):
	def __init__(self, id, value):
		self.id = id
		# self.rank = rank
		self.value = value


# ISSUE HERE, DEADLOCK WITH THIS APPROACH
def async_host_training_using_sharedqueue(host_id, exec_signals, exec_results, federation_lock, federation_weights_queue):


	assert isinstance(exec_signals, AsynchronousExecutionSignals)
	assert isinstance(federation_weights_queue, Queue)

	total_epochs = 10
	total_batches = 2
	host_result = FederationExecutionRes()
	runnable_epochs = 0
	for epoch_id in range(total_epochs):
		for batch_id in range(total_batches):
			# Check for terminating condition
			if exec_signals.is_target_signal_reached():
				federation_weights_queue.put(qObject(host_id, host_id))
				print(host_id, 'First Waiting for Queue to be processed')
				federation_weights_queue.join()

			time.sleep((host_id+1)*0.05)
			host_result.completed_batches += 1

		host_result.completed_epochs += 1
		runnable_epochs += 1
		host_result.latest_weights = [] # update host's weights with its latest trained weights
		exec_results[host_id] = host_result

		if exec_signals.is_target_epochs_reached(runnable_epochs):
			runnable_epochs = 0
			print(host_id, 'Target Reached')

			while exec_signals.is_target_signal_reached():
				print(host_id, 'Second Waiting for Queue to be processed')
				federation_weights_queue.put(qObject(host_id, host_id))
				federation_weights_queue.join()

			print("Host:", host_id, federation_lock.locked())

			# TODO Deadlock Problem here. Need better synchronization.
			# TODO Choice 1:
			# TODO Choice 2.1: Let every thread run without blocking and access whoever received the call
			# TODO Choice 2.2: Remove `while not federation_queue.full()` loop and write down community weight,
			# TODO Choice 2.3: in essence instead of blocking all learners block only those who received the call, inconsistent weights - stale gradients
			# tell everyone to share their weights
			with federation_lock:
				print(host_id, 'Acquired the Lock')
				exec_signals.set_target_signal()

				# Wait for queue to be filled up
				# federation_weights_queue.put(qObject(host_id, host_id))
				# while not federation_weights_queue.full():
				# 	time.sleep(0.001)
				while federation_weights_queue.qsize() < federation_weights_queue.maxsize - 1:
					time.sleep(0.1)
				# while federation_queue.qsize() < 0.3*LEARNERS:
				# 	time.sleep(0.1)

				# Wait for queue to be processed
				res = [qObject(host_id, host_id)]
				while not federation_weights_queue.empty():
					item = federation_weights_queue.get()
					federation_weights_queue.task_done()
					res.append(item)

				# Process Result ~ res
				print(host_id, [item.id for item in res])
				print(host_id, 'Resetting the target')
				exec_signals.reset_target_signal()

				# for res_i in res:
				# 	if res_i.id != host_id:
				# 		federation_weights_queue.put(res_i)

				# for num in range(len(res)):
				# 	federation_weights_queue.task_done()

				print("************ \n\n")



if __name__=='__main__':

	for federation_round in range(0,2):

		federation_lock = threading.Lock()
		federation_condition_variable = threading.Condition(federation_lock)
		community_queue = Queue(maxsize=LEARNERS)

		fedtraining_threads = list()
		exec_signals = AsynchronousExecutionSignals(target_learners=3, target_epochs=2, target_accuracy=0.95)
		exec_results = dict()

		for host_id in range(0, LEARNERS):
			t = threading.Thread(target=async_host_training_using_sharedqueue,
								 name='host_training_{}'.format(host_id),
								 args=[host_id, exec_signals, exec_results, federation_lock, community_queue],
								 daemon=True)
			fedtraining_threads.append(t)

		for p in fedtraining_threads:
			p.start()


		for p in fedtraining_threads:
			p.join()

		print("\n")
		print(exec_signals)
		print(exec_results)