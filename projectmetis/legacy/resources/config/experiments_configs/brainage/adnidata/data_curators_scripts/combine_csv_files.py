import csv
import glob
import os

# for filename in glob.glob("*.csv"):
clients_num = 8
for idx in range(1, clients_num+1):
	new_lines = []
	subject_id = -1
	csv_attributes = []
	for filename in ["homogeneous_datasize_iid_x8clients/with_validation/train_{}.csv".format(idx),
					 "homogeneous_datasize_iid_x8clients/with_validation/valid_{}.csv".format(idx)]:
		with open(filename, 'r') as f:
			lines = csv.DictReader(f)
			csv_attributes = lines.fieldnames
			for line in lines:
				subject_id += 1
				new_line = [subject_id, line['eid'], line['age_at_scan'], line['9dof_2mm_vol'], line['bin']]
				new_lines.append(new_line)

	with open('train_{}.csv'.format(idx), mode='w+') as fileout:
		csvwriter = csv.writer(fileout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csvwriter.writerow(csv_attributes)
		for line in new_lines:
			csvwriter.writerow(line)
