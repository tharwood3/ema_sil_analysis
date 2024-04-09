from IPython.display import display
from ipywidgets import widgets, Layout, Output
import pandas as pd
import os

class Dashboard():
    def __init__(self, compound_data, output_path, polarity, compound_idx=0, plot_size=(1200, 800)):
        
        self.compound_data = compound_data
        self.compound_idx = compound_idx
        self.polarity = polarity
        self.plot_size = plot_size
        self.output_path = output_path
        
        self.widget_out = Output()
        
        self.display_new_compound()
        
    def _init_buttons(self):
        
        self.previous_button = widgets.Button(
            description='Previous Compound',
            disabled=False,
            button_style='')

        self.next_button = widgets.Button(
            description='Next Compound',
            disabled=False,
            button_style='')
        
        self.save_button = widgets.Button(
            description='Save Selections',
            disabled=False,
            button_style='')
        
        self.previous_button.on_click(self.on_previous_button_clicked)
        self.next_button.on_click(self.on_next_button_clicked)
        self.save_button.on_click(self.on_save_button_clicked)
        
    def _init_checkboxes(self):
        item_layout = Layout(flex='1 1 auto', width='auto')
        
        checkboxes = []
        for label in self.compound_data[self.compound_idx]['all_m_signals']:
            
            removed = False
            if "Remove " + label in self.compound_data[self.compound_idx]['remove_m_signals']:
                removed = True
            
            checkboxes.append(widgets.Checkbox(value=removed, description="Remove " + label, 
                                               layout=item_layout))
            
        remove_all = False
        if "Remove All" in self.compound_data[self.compound_idx]['remove_m_signals']:
            remove_all = True
            
        checkboxes.append(widgets.Checkbox(value=remove_all, description="Remove All", 
                                           layout=item_layout))
        
        self.checkboxes = checkboxes
        
    def _init_plot_image(self):
        img1 = open(self.compound_data[self.compound_idx]['plot_path'], 'rb').read()
        wi1 = widgets.Image(value=img1, format='png', width=self.plot_size[0], height=self.plot_size[1])
        
        self.plot_image = wi1
        
    def _clear_display(self):
        self.widget_out.clear_output(wait=True)
        
    def display_new_compound(self):
        
        self._init_plot_image()
        self._init_checkboxes()
        self._init_buttons()
        
        checkboxes_vbox = widgets.VBox(children=self.checkboxes)
        buttons_hbox = widgets.HBox([self.previous_button, self.next_button, self.save_button])
        
        image_by_checkboxes = widgets.HBox([self.plot_image, checkboxes_vbox])
    
        with self.widget_out:
            display(image_by_checkboxes)
            display(buttons_hbox)
            
        display(self.widget_out)
        
    def _update_signals_to_remove(self):
        selected_data = []
        for i in range(0, len(self.checkboxes)):
            if self.checkboxes[i].value == True:
                selected_data = selected_data + [self.checkboxes[i].description]
                    
        self.compound_data[self.compound_idx].update({'remove_m_signals':selected_data})
        
    def _increase_compound_idx(self):
        if self.compound_idx + 1 <= (len(self.compound_data) - 1):
            self.compound_idx += 1
            
    def _decrease_compound_idx(self):
        if self.compound_idx - 1 >= 0:
            self.compound_idx -= 1
            
    def _export_selections(self):
        compound_data_df = pd.DataFrame(self.compound_data)
        compound_data_df['remove_entry'] = compound_data_df['remove_m_signals'].apply(lambda x: True if "Remove All" in x else False)
        
        compound_data_df.to_csv(os.path.join(self.output_path, "{}_gui_selection_data.csv".format(self.polarity)))

    def on_previous_button_clicked(self, b):
        self._update_signals_to_remove()
        self._clear_display()
        self._decrease_compound_idx()
        self.display_new_compound()
            
    def on_next_button_clicked(self, b):
        self._update_signals_to_remove()
        self._clear_display()
        self._increase_compound_idx()
        self.display_new_compound()
        
    def on_save_button_clicked(self, b):
        self._update_signals_to_remove()
        self._export_selections()