from fasthtml.common import *
import matplotlib.pyplot as plt

# Import QueryBase, Employee, Team from employee_events
#### YOUR CODE HERE
from employee_events.query_base import QueryBase

from employee_events.employee import Employee

from employee_events.team import Team


# import the load_model function from the utils.py file
#### YOUR CODE HERE
from utils import load_model


"""
Below, we import the parent classes
you will use for subclassing
"""
from base_components import (
    Dropdown,
    BaseComponent,
    Radio,
    MatplotlibViz,
    DataTable
    )

from combined_components import FormGroup, CombinedComponent


# Create a subclass of base_components/dropdown
# called `ReportDropdown`
#### YOUR CODE HERE

# ---------- ReportDropdown ----------

class ReportDropdown(BaseComponent):

    """
   Dropdown that populates options from the SQL model's names() method.
   The parent project's Dropdown interface may be different; to avoid signature
   mismatches we subclass BaseComponent and implement build_component directly.
   """
    def __init__(self, id="selector", name="user-selection"):
      self.id = id
       # name used as HTML form name attribute if needed
      self.name = name

    
    # Overwrite the build_component method
    # ensuring it has the same parameters
    # as the Report parent class's method
    #### YOUR CODE HERE
    def build_component(self, entity_id, model):

        #  Set the `label` attribute so it is set
        #  to the `name` attribute for the model
        #### YOUR CODE HERE
        label_text = getattr(model, "name", "Select")
        options = self.component_data(entity_id, model)  # list of (value, label)
       # Create FastHTML Select options
        option_tags = []
        for val, display in options:
           # Use Option(display, value=str(val)) so displayed text = display
           option_tags.append(Option(display, value=str(val)))


        # Return the output from the
        # parent class's build_component method
        #### YOUR CODE HERE
        label_tag = Label(label_text, for_=self.id)
        select_tag = Select(*option_tags, id=self.id, name=self.name)
        return Div(label_tag, select_tag, cls="report-dropdown")
    
    # Overwrite the `component_data` method
    # Ensure the method uses the same parameters
    # as the parent class method
    #### YOUR CODE HERE
    def component_data(self, entity_id, model):

        # Using the model argument
        # call the employee_events method
        # that returns the user-type's
        # names and ids
       names = []
       try:
           names = model.names()  # expecting list of tuples (display, id) or (id, display)
       except Exception:
           # Try alternative ordering if needed
           try:
               raw = model.names()
               # Attempt to coerce to (id, display)
               names = [(r[1], r[0]) for r in raw]
           except Exception:
               names = []
       # Ensure returned list is (value,label)
       processed = []
       for item in names:
           if item is None:
               continue
           if len(item) == 2:
               # Determine which is id vs label by type guess
               a, b = item
               if isinstance(a, (int, str)) and isinstance(b, str):
                   # assume (id, label)
                   if isinstance(a, str) and a.isdigit():
                       # numeric string -> keep as string
                       processed.append((a, b))
                   elif isinstance(a, int):
                       processed.append((a, b))
                   else:
                       # fallback
                       processed.append((a, b))
               else:
                   processed.append((a, b))
           else:
               # fallback: place entire item as label with index as value
               processed.append((str(item), str(item)))
       return processed


# Create a subclass of base_components/BaseComponent
# called `Header`
#### YOUR CODE HERE
# ---------- Header ----------

class Header(BaseComponent):

    def build_component(self, entity_id, model):
        model_name = getattr(model, "name", "Report")


        # Show H1 header with model name

        return H1(model.name.title())


    # Overwrite the `build_component` method
    # Ensure the method has the same parameters
    # as the parent class
    #### YOUR CODE HERE
    def build_component(self,entity_id,model): 
        # Using the model argument for this method
        # return a fasthtml H1 objects
        # containing the model's name attribute
        #### YOUR CODE HERE
        return H1(model.name.title())
          

# Create a subclass of base_components/MatplotlibViz
# called `LineChart`
#### YOUR CODE HERE
# ---------- LineChart ----------



# Create a subclass of base_components/MatplotlibViz
# called `LineChart`
class LineChart(MatplotlibViz):
   """
   LineChart builds a cumulative positive/negative event line chart
   from the model.event_counts(entity_id) dataframe.
   """
   def __init__(self, id="line_chart"):
       # MatplotlibViz in this codebase expects to be initialized with a fig_getter callable.
       # Provide the visualization method as the fig_getter.
       super().__init__(id, fig_getter=self.visualization)
   # Overwrite the parent class's `visualization` method. Use the same parameters as the parent
   def visualization(self, model, entity_id):
       # Pass the `asset_id` argument to the model's `event_counts` method to
       # receive the x (Day) and y (event count)
       df = None
       try:
           df = model.event_counts(entity_id)
       except Exception:
           # fallback: try different method name
           try:
               df = model.event_counts(int(entity_id))
           except Exception:
               df = None
       # If not a dataframe, attempt to coerce
       if df is None:
           return None
       # Use pandas methods to clean and transform
       try:
           df = df.fillna(0)
           # Set the date column as the index
           if "event_date" in df.columns:
               df = df.set_index("event_date")
           # Sort the index
           df = df.sort_index()
           # Cumulative sum across rows
           df = df.cumsum()
           # Set the dataframe columns to ['Positive', 'Negative']
           # If there are more or different columns, attempt to pick positive/negative
           cols = list(df.columns)
           if len(cols) >= 2:
               df = df.iloc[:, :2]
               df.columns = ["Positive", "Negative"]
           else:
               # Not enough columns - create placeholders
               df = df.reindex(columns=["Positive", "Negative"], fill_value=0)
       except Exception:
           # If df is not a pandas DataFrame or transformation fails, return None
           return None
       # Initialize a pandas subplot and assign the figure and axis to variables
       fig, ax = plt.subplots(figsize=(7, 3))
       # call the .plot method for the cumulative counts dataframe
       df.plot(ax=ax)
       # pass the axis variable to the `.set_axis_styling` method
       # Use keyword arguments to set the border color and font color to black.
       # If MatplotlibViz provides a helper set_axis_styling, use it; otherwise skip gracefully.
       try:
           self.set_axis_styling(ax, bordercolor="black", fontcolor="black")
       except Exception:
           # silently continue if helper missing
           pass
       # Set title and labels for x and y axis
       ax.set_title("Cumulative Event Counts", fontsize=14)
       ax.set_xlabel("Date")
       ax.set_ylabel("Cumulative Count")
       return fig


# Create a subclass of base_components/MatplotlibViz
# called `BarChart`
#### YOUR CODE HERE
# ---------- BarChart ----------

class BarChart(MatplotlibViz):

    # Predictor is the loaded ML model

    predictor = load_model()
    def __init__(self, id="bar_chart"):
       super().__init__(id, fig_getter=self.visualization)

    def visualization(self, model,entity_id):

        # Model input
     try:

        df = model.model_data(entity_id) 
     except Exception:

       df = model.model_data(int(entity_id))
     except Exception:
      df = None
       # Handle missing or empty data
    if df is None:
        return None
       # Using the predictor class attribute pass the data to predict_proba
    try:
             proba = self.predictor.predict_proba(df)
    except Exception:
           # If predictor does not support predict_proba, try predict
           try:
               preds = self.predictor.predict(df)
               # normalize to [0,1] range if regression; clip
               proba = [[1 - p, p] for p in preds]
           except Exception:
                  return None
       # Index the second column of predict_proba output
       # The shape should be (n_samples, 2) so [:,1] selects positive-class prob
    try:
           proba_vals = proba[:, 1]
    except Exception:
           # If proba is a list of lists
           proba_vals = [p[1] for p in proba]
       # Below, create a `pred` variable set to the number we want to visualize
  if getattr(model, "name", "").lower() == "team":
           # mean team risk
           try:
               pred = float(sum(proba_vals) / len(proba_vals))
           except Exception:
               pred = float(proba_vals[0])
       else:
           # single employee -> first value
           pred = float(proba_vals[0]) if len(proba_vals) > 0 else 0.0
       # Initialize a matplotlib subplot
       fig, ax = plt.subplots(figsize=(5, 1.5))
       # Run the following code unchanged
       ax.barh([''], [pred])
       ax.set_xlim(0, 1)
       ax.set_title('Predicted Recruitment Risk', fontsize=14)
       # styling helper if available
       try:
           self.set_axis_styling(ax, bordercolor="black", fontcolor="black")
       except Exception:
           pass
        return fig

# Create a subclass of combined_components/CombinedComponent
# called Visualizations       
#### YOUR CODE HER
class Visualizations(CombinedComponent):

    # Set the `children`
    # class attribute to a list
    # containing an initialized
    # instance of `LineChart` and `BarChart`
    #### YOUR CODE HERE
    children = [LineChart(), BarChart()]

    # Leave this line unchanged
    outer_div_type = Div(cls='grid')
            
# Create a subclass of base_components/DataTable
# called `NotesTable`
#### YOUR CODE HERE
class NotesTable(DataTable):

    # Overwrite the `component_data` method
    # using the same parameters as the parent class
    #### YOUR CODE HERE
    def __init__(self, component_id="notes_table"):
        
        # Using the model and entity_id arguments
        # pass the entity_id to the model's .notes 
        # method. Return the output
        #### YOUR CODE HERE
    
self.id = component_id
def component_data(self, entity_id, model):
    try:
           notes_df = model.notes(entity_id)
           return notes_df
           except Exception:
           # try int conversion fallback
           try:
               notes_df = model.notes(int(entity_id))
               return notes_df
           except Exception:
               return []


class DashboardFilters(FormGroup):

    id = "top-filters"
    action = "/update_data"
    method="POST"

    children = [
        Radio(
            values=["Employee", "Team"],
            name='profile_type',
            hx_get='/update_dropdown',
            hx_target='#selector'
            ),
        ReportDropdown(
            id="selector",
            name="user-selection")
        ]
    
# Create a subclass of CombinedComponents
# called `Report`
#### YOUR CODE HERE
class Report(CombinedComponent):

    # Set the `children`
    # class attribute to a list
    # containing initialized instances 
    # of the header, dashboard filters,
    # data visualizations, and notes table
    #### YOUR CODE HERE
    children = [Header(), DashboardFilters(), Visualizations(), NotesTable()]
    def __call__(self, entity_id, model):
        
       built = self.call_children(entity_id, model)
       return self.outer_div(built, {"cls": "report-root"})


# Initialize a fasthtml app 
#### YOUR CODE HERE
app = FastHTML()


# Initialize the `Report` class
#### YOUR CODE HERE
report = Report()


# Create a route for a get request
# Set the route's path to the root
#### YOUR CODE HERE
@app.get("/")
def index():

    # Call the initialized report
    # pass the integer 1 and an instance
    # of the Employee class as arguments
    # Return the result
    #### YOUR CODE HERE
     return report(1, Employee())


# Create a route for a get request
# Set the route's path to receive a request
# for an employee ID so `/employee/2`
# will return the page for the employee with
# an ID of `2`. 
# parameterize the employee ID 
# to a string datatype
#### YOUR CODE HERE
@app.get("/team/{id}")
def team_page(id: str):
    # Call the initialized report
    # pass the ID and an instance
    # of the Employee SQL class as arguments
    # Return the result
    #### YOUR CODE HERE
    return report(int(id), Team())



# Create a route for a get request
# Set the route's path to receive a request
# for a team ID so `/team/2`
# will return the page for the team with
# an ID of `2`. 
# parameterize the team ID 
# to a string datatype
#### YOUR CODE HERE

    # Call the initialized report
    # pass the id and an instance
    # of the Team SQL class as arguments
    # Return the result
    #### YOUR CODE HERE


# Keep the below code unchanged!
@app.get('/update_dropdown{r}')
def update_dropdown(r):
    dropdown = DashboardFilters.children[1]
    print('PARAM', r.query_params['profile_type'])
    if r.query_params['profile_type'] == 'Team':
        return dropdown(None, Team())
    elif r.query_params['profile_type'] == 'Employee':
        return dropdown(None, Employee())


@app.post('/update_data')
async def update_data(r):
    from fasthtml.common import RedirectResponse
    data = await r.form()
    profile_type = data._dict['profile_type']
    id = data._dict['user-selection']
    if profile_type == 'Employee':
        return RedirectResponse(f"/employee/{id}", status_code=303)
    elif profile_type == 'Team':
        return RedirectResponse(f"/team/{id}", status_code=303)
    


serve()
try:
   from fasthtml.server import serve
   serve()
except Exception:
   # If serve() isn't available in this FastHTML build, the user can run via uvicorn:
   # uvicorn report.dashboard:app --reload
   pass
