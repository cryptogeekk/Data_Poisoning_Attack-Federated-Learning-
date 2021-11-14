class Client:
    
    def __init__(self,dataset, dataset_x, dataset_y, epoch_number, learning_rate,weights,batch):
        self.dataset = dataset
        self.dataset_x=dataset_x
        self.dataset_y=dataset_y
        self.epoch_number=epoch_number
        self.learning_rate=learning_rate
        self.weights=weights
        self.batch=batch
        
    def train(self): 
        from utils import get_model
        print("client getting model")
        model = get_model(self.dataset)

        model.set_weights(self.weights)
        
        model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
        history=model.fit(self.dataset_x, self.dataset_y,epochs=self.epoch_number,batch_size=self.batch) 

        output_weight=model.get_weights()
        
        return output_weight
        
        
        

        



    