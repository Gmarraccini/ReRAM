import tensorflow as tf
import numpy as np
import pylab as pl
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay



def quantize(matr):
  q_matr = np.array(matr/tf.reduce_max(tf.math.abs(matr)))
  # print(matr.shape)
  # print(len(q_matr.shape))
  if len(matr.shape) == 2:
    for r in range(matr.shape[0]):
      for c in range(matr.shape[1]):
        if matr[r][c] <= 0:
          q_matr[r][c] = q_matr[r][c]*(2**(bit-1))
        else:
          q_matr[r][c] = q_matr[r][c]*(2**(bit-1)-1)
    q_matr = np.array(q_matr, dtype=np.int32)
  else:
    for r in range(matr.shape[0]):
      if matr[r] <= 0:
        q_matr[r] = q_matr[r]*(2**(bit-1))
      else:
        q_matr[r] = q_matr[r]*(2**(bit-1)-1)
    q_matr = np.array(q_matr, dtype=np.int32)
  return q_matr

tf.keras.backend.clear_session()
# tf.keras.utils.set_random_seed(654)
# tf.config.experimental.enable_op_determinism()

'''
#Old dataset
zer = np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]])
one = np.array([[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]])
two = np.array([[1,1,1,1],[0,0,0,1],[1,1,1,1],[1,0,0,0],[1,1,1,1]])
tre = np.array([[1,1,1,1],[0,0,0,1],[0,1,1,1],[0,0,0,1],[1,1,1,1]])
fou = np.array([[1,0,0,1],[1,0,0,1],[1,1,1,1],[0,0,0,1],[0,0,0,1]])
fiv = np.array([[1,1,1,1],[1,0,0,0],[1,1,1,1],[0,0,0,1],[1,1,1,1]])
six = np.array([[1,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,1],[1,1,1,1]])
sev = np.array([[1,1,1,1],[0,0,0,1],[0,0,1,0],[0,0,1,0],[0,0,1,0]])
eig = np.array([[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,1,1,1]])
nin = np.array([[1,1,1,1],[1,0,0,1],[1,1,1,1],[0,0,0,1],[0,0,0,1]])

'''
#Fancy dataset
zer = np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]])
one = np.array([[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,1,1,1]])
two = np.array([[1,1,1,1],[0,0,0,1],[1,1,1,1],[1,0,0,0],[1,1,1,1]])
tre = np.array([[1,1,1,1],[0,0,0,1],[0,1,1,1],[0,0,0,1],[1,1,1,1]])
fou = np.array([[0,0,1,1],[0,1,0,1],[1,0,0,1],[1,1,1,1],[0,0,0,1]])
fiv = np.array([[1,1,1,1],[1,0,0,0],[1,1,1,1],[0,0,0,1],[1,1,1,1]])
six = np.array([[1,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,1],[1,1,1,1]])
sev = np.array([[1,1,1,1],[0,0,0,1],[0,0,1,0],[0,0,1,0],[0,0,1,0]])
eig = np.array([[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1],[0,1,1,0]])
nin = np.array([[1,1,1,1],[1,0,0,1],[1,1,1,1],[0,0,0,1],[0,0,0,1]])

labels = np.zeros((10,10))
for i in range(10):
  labels[i,i] = 1

ds_train = [zer.flatten(),one.flatten(),two.flatten(),tre.flatten(),fou.flatten(),fiv.flatten(),six.flatten(),sev.flatten(),eig.flatten(),nin.flatten()]
corrupt_train = ds_train
ds_train = tf.convert_to_tensor(ds_train)
ds_train = tf.cast(ds_train, tf.float32)
labels = tf.convert_to_tensor(labels)
labels = tf.cast(labels, tf.float32)



loss_func = tf.keras.losses.CategoricalCrossentropy()
cmd = input('>Load model? (y/n): ')
if cmd == 'n':
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20,)),
        tf.keras.layers.Dense(20, activation='relu', name='I1'),
        tf.keras.layers.Dense(20, activation='relu', name='H2'),
        # tf.keras.layers.Dense(30, activation='relu', name='H2'),
        tf.keras.layers.Dense(10, activation='softmax', name='O3')
    ])
    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate = 1e-3),
        loss= loss_func,
        #metrics=['MeanAbsoluteError']
        metrics=['acc'])
    
    model.fit(ds_train, labels, epochs = 600 )
    
    modelname = 'Mk8I20H20B.keras'
    model.save(modelname)
    print(f'>Model saved as {modelname}')
else:
    cmd = input('>Model file: ')
    modelname = cmd
    model = tf.keras.models.load_model(cmd)
model.summary()

#-------------------------Kinda quantization
print('>Quantizing...')
bit = 3
for lay in model.layers:
    if lay.name == 'I1':
      w_I = lay.get_weights()[0]
      b_I = lay.get_weights()[1]
    elif lay.name == 'H2':
      w_h = lay.get_weights()[0]
      b_h = lay.get_weights()[1]
    else:
      w_o = lay.get_weights()[0]
      b_o = lay.get_weights()[1]

qw_I = quantize(w_I)
qb_I = quantize(b_I)

qw_h = quantize(w_h)
qb_h = quantize(b_h)

qw_o = quantize(w_o)
qb_o = quantize(b_o)

# for pippolo in range(len(w_I)):
  # pl.scatter(w_I[pippolo], qw_I[pippolo])
# pl.xlabel('float weight')
# pl.ylabel('quantized weight')
# pl.title(f'Layer1 {bit} bit quantized')
# pl.show()


for layer in model.layers:
  if layer.name == 'I1':
    layer.set_weights([qw_I,qb_I])
  elif layer.name == 'H2':
    layer.set_weights([qw_h,qb_h])
  else:
    layer.set_weights([qw_o,qb_o])
model.save_weights('intNEW_TRAINED.weights.h5', overwrite=True)

model.summary()

for i,layer in enumerate(model.layers):
    weights, biases = layer.get_weights()  # Get weights and biases
    print(f"Layer: {layer.name}")
    print(f"Weights:\n{weights}")
    print(f"Biases:\n{biases}\n")


#---------------------Inference
for r in range(len(ds_train)):
  corrupt = np.random.randint(20)
  corrupt_train[r][corrupt] = corrupt_train[r][corrupt] ^ 1
corrupt_train = tf.convert_to_tensor(corrupt_train)
corrupt_train = tf.cast(corrupt_train, tf.float32)

cmd = input('>Corrupted test? (y/n): ')
if cmd == 'n':
    predictions = model.predict(ds_train)
    for id,row in enumerate(ds_train):
        print(f'{id}: \n{row[:4]}\n{row[4:8]}\n{row[8:12]}\n{row[12:16]}\n{row[16:20]}')
else:
    predictions = model.predict(corrupt_train)
    for id,row in enumerate(corrupt_train):
        print(f'{id}: \n{row[:4]}\n{row[4:8]}\n{row[8:12]}\n{row[12:16]}\n{row[16:20]}')


y_pred = np.argmax(predictions, axis=1)

# Calculate confusion matrix
y_true = [0,1,2,3,4,5,6,7,8,9]
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, zero_division = 0))


fig, ax = pl.subplots(figsize=(8, 8))  # Customize the size if needed
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp = cmd.plot(cmap=pl.cm.Blues, ax=ax)

# Increase font size for labels and values
for text in disp.ax_.texts:
    text.set_fontsize(20)  # Adjust this number to your preference

# Optionally increase axis label sizes too
disp.ax_.set_xlabel('Predicted label', fontsize=24)
disp.ax_.set_ylabel('True label', fontsize=24)
disp.ax_.tick_params(axis='both', labelsize=24)

if cmd == 'y':
    model += 'Corr'
fig.savefig(f'{modelname}.png', dpi = 200)

# pl.title("Confusion Matrix", fontsize=18)
pl.show()

# exit()

#-----------------------ARRAY VIRTUALE
class memristor():
    def __init__(self):
        self.LRS = np.random.lognormal(LRS_avg, LRS_ds)
        self.HRS = 10**np.random.normal(HRS_avg, HRS_ds)
        # self.Vset = np.random.lognormal(Vset_avg, Vset_ds)
        self.Vset = np.random.lognormal(Vset_avg, Vset_avg)
        self.Vrst = -np.random.lognormal(Vrst_avg, Vrst_ds)
        self.state = self.HRS

    def app_V(self, V):
        if V > self.Vset:
            self.state = self.LRS
            return V / self.state
        elif V < self.Vrst:
            self.state = self.HRS
            return V / self.state
        else:
            return V / self.state

    def read(self):
        return self.state

    def reset(self):
        self.state = self.HRS

    def get_params(self):
      if self.Vset > 5:
        print('FAULT')
      print(f'LRS: {self.LRS}')
      print(f'HRS: {self.HRS}')
      print(f'Vset: {self.Vset}')
      print(f'Vrst: {self.Vrst}')


def resetCA(w_matr, b_vect):
  for r in range(len(w_matr)):
    for c in range(len(w_matr[0])):
      w_matr[r][c].reset()
  for r in range(len(b_vect)):
    b_vect[r].reset()

def updateCA(w_matr, b_vect, MRW, MRB):
  for r in range(w_matr.shape[0]):
    for c in range(w_matr.shape[1]):
        if w_matr[r][c] < 0:
            # num = (w_matr[r][c]^0b111) + 1
            num = (1 << bit) + w_matr[r][c]
        else:
            num = w_matr[r][c]
        # print(num)
        # for i,b in enumerate(bin(num)[::-1]):
          # if b != 'b' :
            # if b == '1':
              # MRW[r][c*bit + (bit-1) - i].app_V(Vset)
            # else:
              # MRW[r][c*bit + (bit-1) - i].app_V(Vrst)
        bitstring = f'{num:0{bit}b}'  # format to binary with fixed bit width
        for i, b in enumerate(bitstring):
            if b == '1':
                MRW[r][c * bit + i].app_V(Vset)
            else:
                MRW[r][c * bit + i].app_V(Vrst)
            # else:
                # break
    for r in range(b_vect.shape[0]):
        if b_vect[r] < 0:
            # bias = (b_vect[r]^0b111) + 1
            bias = (1 << bit) + b_vect[r]
        else:
            bias = b_vect[r]
      # for i,b in enumerate(bin(bias)[::-1]):
        bitstring_w = f'{num:0{bit}b}'
        for i,b in enumerate(bitstring):
            if b == '1':
                MRB[bit*r + bit-1 - i].app_V(Vset)
            else:
                MRB[bit*r + bit-1 - i].app_V(Vrst)
        # else:
            # break
# print(len(IN_Wca), len(IN_Wca[0]))

def ConvertInt(w_matr):
  int_matr = np.zeros((len(w_matr), int(len(w_matr[0])/bit)))
  curr_matr = np.zeros((len(w_matr), int(len(w_matr[0])/bit)))

  for r in range(len(w_matr)):
    for c in range(len(int_matr[0])):
      bitstring = ''
      #------Correnti prese facendo Vb0/Rb0 + Vb1/Rb1 + Vb2/Rb2
      curr_matr[r][c] = (w_matr[r][bit*c].app_V(1e-3)+w_matr[r][bit*c+1].app_V(2e-3)+w_matr[r][bit*c+2].app_V(4e-3))
      # print(w_matr[r][c].read())
      for i in range(bit):
          if w_matr[r][c*bit -1 -i] < Rthr:
            bitstring += str(w_matr[])
      value = int(bitstring, 2)
      if value >= (1 << (len(bitstring) - 1)): 
          value -= (1 << len(bitstring)) 
      int_matr[r][c] = value

  return int_matr, curr_matr

Vset = 5
Vrst = -5

Vset_avg = .20177
Vset_ds = .776
Vrst_avg = -0.00248
Vrst_ds = 0.810

LRS_avg = 6.253
LRS_ds = 1.041  #Lognormal
HRS_avg = 6.041 #Distribuzione del valore in logaritmo
HRS_ds = .769

# print(bit*qw_I.shape[1])

IN_Wca = [[memristor() for _ in range(bit*qw_I.shape[1])] for __ in range(qw_I.shape[0])]
IN_Bca = [memristor() for _ in range(bit*qb_I.shape[0])]
H1_Wca = [[memristor() for _ in range(bit*qw_h.shape[1])] for __ in range(qw_h.shape[0])]
H1_Bca = [memristor() for _ in range(bit*qb_h.shape[0])]

O_Wca = [[memristor() for _ in range(bit*qw_o.shape[1])] for __ in range(qw_o.shape[0])]
O_Bca = [memristor() for _ in range(bit*qb_o.shape[0])]

for r in IN_Wca:
  for c in range(len(r)):
    print(r[c].get_params())
    print('__________') #By default writes FAULT if VSET > 5

# resetCA(IN_Wca, IN_Bca)
# resetCA(H1_Wca, H1_Bca)
# resetCA(O_Wca, O_Bca)

updateCA(qw_I, qb_I,IN_Wca, IN_Bca)
print('L1 done')

updateCA(qw_h, qb_h,H1_Wca, H1_Bca)

updateCA(qw_o, qb_o,O_Wca, O_Bca)

# for r in IN_Wca:
  # for c in range(len(r)):
    # print(round(r[c].read(),1), end = '  ')
  # print('\n\n')



# print(len(IN_Wca[0]))
Rthr = 1e4
reconv_wIN, i_matrIN = ConvertInt(IN_Wca)
reconv_wH1, i_matrH1 = ConvertInt(H1_Wca)
reconv_wO, i_matrO  = ConvertInt(O_Wca)

print(reconv_wIN - qw_I)
print(reconv_wH1 - qw_h)
print(reconv_wO - qw_o)