from background_scene_fits import background_scene
class bypass_datalayer:
    
    def setup(self,bottom,top,pytorch=None):

        #--- read parameters from `self.param_str`
        self.params = eval(self.param_str)


        #--- Create the background_scene object
        self.background_scene = background_scene(self.params);

        return len(self.background_scene.back_images_list)
    
    def forward(self,bottom,top,pytorch=None,pytorch_index=0):
                
        img = self.background_scene.generate((pytorch_index,None)) 
        return img

