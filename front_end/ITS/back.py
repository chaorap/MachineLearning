from flask import Flask, render_template,request,jsonify
import json

app = Flask(__name__, static_url_path='')

@app.route('/')
def ind():
	return app.send_static_file('index.html')

@app.route('/ListPowerModule',methods=['POST'])
def ListPowerModule():
	t = [
		{
			'PM-Name': "PM1",
			'PM-Path': "dwa.23213.csadwa",
			'PM-Size': "6E/4",
			"PM-Current":"300",
			"PM-EnvSensor-Path":"PM1.Sensors.Environment",
			"PM-ContSensor-Path":"PM1.Sensors.OutgoingL1"
		},
		{
			'PM-Name': "PM55",
			'PM-Path': "pok.2312.06546",
			'PM-Size': "64E",
			"PM-Current":"700",
			"PM-EnvSensor-Path":"PM55.Sensors.Environment",
			"PM-ContSensor-Path":"PM1.Sensors.OutgoingL1_1"
		},
	]
	return jsonify(t)

@app.route('/ModelStatus',methods=['POST'])
def ModelStatus():
	t = [
		{'ModelGuid': "3213213213dawdwa",
		'ModelDescription': "mmmdwadwa",
		'ModelStatus': "Ongoing"},
		{'ModelGuid': "11",
		'ModelDescription': "22",
		'ModelStatus': "33"}
	]
	return jsonify(t)

@app.route('/a1', methods=['POST'])
def register():

	#如果传的是普通的key/value的表单form形式类型
	#如果是post的用 request.form.get("key")
	#如果是get的用 request.args.get("key")

	print(request.form.get("name"))
	print(request.form.get("no"))

	t = {
		'a': 1,
		'b': 2,
		'c': [3, 4, 5]
	}
	return jsonify(t)

    # data = request.json
    # data_name = data.get('name')
	# data_no = data.get('name')

	# f1 = request.form.get('f1')
	# f2 = request.form.get('f2')	
	# f3 = request.form.get('f3')	

	# data_name = request.args.get("name")
	# data_no = request.args.get("no")
	# print("---------------------		%s	%s	%s"%(f1,f2,f3))

	# retdict['data'] = [username, password]
	# return json.dumps(retdict,ensure_ascii=False)

	# print(request.data)
	# print(request.get_data())

	# jdata =  json.loads(data)

	# #data = json.loads(request.form.get('data'))
	# username = data['username']
	# password = data['password']
	# print (username)
	# print (password)
	# return "46575"
	# data = json.loads(request.get_data(as_text=True))
	# # data = request.get_json()
	# print(data)
	# return json.dumps(data,ensure_ascii=False) 

if __name__ == '__main__':
    app.run(port=6088, debug=True)