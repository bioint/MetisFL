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


class SemiSynchronousExecutionSignals(ExecutionInterruptionSignals):

	def __init__(self, target_learners=None, target_epochs=None, target_accuracy=None, target_exec_time_mins=None):
		ExecutionInterruptionSignals.__init__(self, target_learners, target_epochs, target_accuracy, target_exec_time_mins)
		self.fedround_completed_learners = 0
		self.fedround_completed_epochs = 0
		self.fedround_test_accuracy = 0

	def is_semisync_signal_reached(self):
		if self.target_learners is None and self.target_epochs is None:
			return False  # Let the framework proceed with normal execution synchronous execution
		if self.target_epochs is not None: # this gets executed only in semi-synchronous policies
			return self.fedround_completed_epochs >= self.target_epochs
		if self.target_learners is not None: # this gets executed only in semi-synchronous policies
			return self.fedround_completed_learners >= self.target_learners

	def __str__(self):
		return "Finished Learners: {}, Finished Epochs: {}".format(self.fedround_completed_learners, self.fedround_completed_epochs)


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


def sync_host_training(host_id, exec_signals, exec_results, host_lock):

	assert isinstance(exec_signals, SemiSynchronousExecutionSignals)


	total_epochs = 5
	completed_batches = 0
	completed_epochs = 0
	for epoch_id in range(total_epochs):

		current_batch = 0
		while current_batch < 5:

			# Check for terminating condition
			with host_lock:
				if exec_signals.fedround_completed_epochs >= exec_signals.target_epochs:
					exec_results[host_id] = (completed_epochs, completed_batches)
					return

			time.sleep((host_id+1)*0.5) # batch processing simulation
			current_batch += 1
			completed_batches += 1

		completed_epochs += 1

		print("HostID: {}, EpochID: {}, Host Completed Epochs: {}, Framework Execution Signals -> {}".format(host_id, epoch_id, completed_epochs, exec_signals))

		# Update completed epochs and check for terminating condition
		with host_lock:
			exec_signals.fedround_completed_epochs += 1

	with host_lock:
		exec_signals.fedround_completed_learners += 1
		exec_results[host_id] = (completed_epochs, completed_batches)
		return



if __name__=='__main__':

	for federation_round in range(0,2):

		federation_lock = threading.Lock()
		federation_condition_variable = threading.Condition(federation_lock)
		community_queue = Queue(maxsize=LEARNERS)

		fedtraining_threads = list()
		exec_signals = SemiSynchronousExecutionSignals(target_learners=3, target_epochs=2, target_accuracy=0.95)
		exec_results = dict()

		for host_id in range(0, LEARNERS):
			t = threading.Thread(target=sync_host_training,
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