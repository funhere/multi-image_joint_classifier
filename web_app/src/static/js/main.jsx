//import { Button } from 'reactstrap';
//import React from 'react';
const Button = window.Reactstrap.Button;


const Collapse = window.Reactstrap.Collapse;
const Navbar = window.Reactstrap.Navbar;
const NavbarBrand = window.Reactstrap.NavbarBrand;
const Nav = window.Reactstrap.Nav;
const NavItem = window.Reactstrap.NavItem;
const NavLink = window.Reactstrap.NavLink;


const Router = window.ReactRouterDOM.BrowserRouter;
const Route = window.ReactRouterDOM.Route;
const ReactMarkdown = window.ReactMarkdown;

const Form = window.Reactstrap.Form;
const FormGroup = window.Reactstrap.FormGroup;
const Label = window.Reactstrap.Label;
const Input = window.Reactstrap.Input;


const UncontrolledDropdown = window.Reactstrap.UncontrolledDropdown;
const Dropdown = window.Reactstrap.Dropdown;
const DropdownToggle = window.Reactstrap.DropdownToggle;
const DropdownMenu = window.Reactstrap.DropdownMenu;
const DropdownItem = window.Reactstrap.DropdownItem;
const Spinner = window.Reactstrap.Spinner;



const axios = window.axios;

const Select = window.Select;


//import { Button } from 'reactstrap';

// Obtain the root 
const rootElement = document.getElementById('root');


class About extends React.Component {
    //

// Use the render function to return JSX component
    render() {
        return (

            <div>
                <h1>About</h1>
                <ReactMarkdown source={window.APP_CONFIG.about}/>
            </div>
        );
    }
}


// Create a ES6 class component
class MainPage extends React.Component {
    //

    constructor(props) {
        super(props);
        this.state = {
            //file: null,
            file: [],
            predictions: [],
            imageSelected: false,
            url: [],
            isLoading: false,
            selectedOption: null,

        }
    }

    _onFileUpload = (event) => {
        var imgList = []
        var files = event.target.files; //FileList object
        for(var i = 0; i< files.length; i++)
        {
            var f = files[i];
            
            //Only pics
            if(!f.type.match('image'))
                continue;
            imgList[i] = URL.createObjectURL(f);
        }   
            
        
        const arrFiles = Array.from(event.target.files)
        console.log(event.target.files)
        this.setState({
            rawFile: event.target.files,
            //file: URL.createObjectURL(event.target.files[0]),
            file: imgList,
            imageSelected: true
//             rawFile: arrFiles,
//             file: URL.createObjectURL(event.target.files[0]),
//             imageSelected: true
        })
        
//         //preview multiple images before upload
//         var files = event.target.files; //FileList object
//         var output = document.getElementById("result");
//         console.log(this.state.imageSelected)
        
//         var flag_hidden = false
//         if(this.state.imageSelected)
//             flag_hidden = false
//         else
//             flag_hidden = true 
//         console.log(flag_hidden)
            
//         for(var i = 0; i< files.length; i++)
//         {
//             var file = files[i];
                
//             //Only pics
//             if(!file.type.match('image'))
//                 continue;
                
//             var picReader = new FileReader();
                
//             picReader.addEventListener("load",function(event){
                    
//                 var picFile = event.target;
                    
//                 var div = document.createElement("div");
//                 div.innerHTML = "<img src='" + picFile.result + "'"
//                              + " className={'img-preview'} " + " />";
//                 var ss1 = "<img src='"  + "'"
//                              + " className={'img-preview'} " + " hidden={" + flag_hidden + "} />";
//                 console.log(ss1)                
//                 output.insertBefore(div,null);            
                
//             });
                
//             //Read the image
//             picReader.readAsDataURL(file);
//         } 
        
        //rawFile.append(event.target.files[0]),
        //rawFile.append(event.target.files[1]),
        //file.append(URL.createObjectURL(event.target.files[0])),
        //file.append(URL.createObjectURL(event.target.files[1]))
        //arrFiles.forEach((itm, i) => {
        //    rawFile.append(itm),
        //    file.append(URL.createObjectURL(itm))   
        //})
        
    };

    _onUrlChange = (url) => {
        this.state.url = url;
        if ((url.length > 5) && (url.indexOf("http") === 0)) {
            this.setState({
                file: url,
                imageSelected: true
            })
        }
    };

    _clear = async (event) => {
        this.setState({
            file: [],
            imageSelected: false,
            predictions: [],
            rawFile: [],
            url: ""
        })
    };

    _predict = async (event) => {
        this.setState({isLoading: true});
        let resPromise = null;
        var rawFiles = this.state.rawFile
        //if (this.state.rawFile) {
        if (rawFiles) {
            const data = new FormData();
            for(var i = 0; i< rawFiles.length; i++)
            {
                data.append('file', rawFiles[i]);
            } 
//             data.append('file', this.state.rawFile[0]);
//             data.append('file', this.state.rawFile[1]);
//             console.log(this.state.rawFile[0])
//             console.log(this.state.rawFile[1])
            resPromise = axios.post('/api/classify', data);
        } else {
            resPromise = axios.get('/api/classify', {
                params: {
                    url: this.state.file
                }
            });
        }
//         let resPromise = null;
//         if (this.state.rawFile) {
//             const data = new FormData();
//             data.append('file', this.state.rawFile[0]);
//             data.append('file', this.state.rawFile[1]);
//             resPromise = axios.post('/api/classify', data);
//         } else {
//             resPromise = axios.get('/api/classify', {
//                 params: {
//                     url: this.state.file
//                 }
//             });
//         }

        try {
            const res = await resPromise;
            const payload = res.data;
            
            this.setState({predictions: payload, isLoading: false});
            console.log(payload)
        } catch (e) {
            alert(e)
        }
    };

    renderPrediction() {
        const predictions = this.state.predictions || [];
        
        if (predictions.length > 0) {

            const predictionItems = predictions.map((item) =>
                <li>{item.class}   ({item.prob}) </li>
                           
            );

            return (
                <ul>
                    {predictionItems}
                </ul>
            )

        } else {
            return null
        }
    }

    handleChange = (selectedOption) => {
        this.setState({selectedOption});
        console.log(`Option selected:`, selectedOption);
    };

    sampleUrlSelected  = (item) => {
        this._onUrlChange(item.url);
    };
    render() {
        const sampleImages = APP_CONFIG.sampleImages;
        return (
            <div>
                <h2>{APP_CONFIG.description}</h2>

                <Form>

                    <FormGroup id={"upload_button"}>
                        <div>
                            <p>Please upload 2 images</p>
                        </div>
                        <Label for="imageUpload">
                            <Input type="file" name="file" id="imageUpload" accept=".png, .jpg, .jpeg" ref="file"
                                   onChange={this._onFileUpload} multiple/>

                            <span className="btn btn-primary">Upload</span>
                        </Label>
                    </FormGroup>

                    <img src={this.state.file[0]} className={"img-preview"} hidden={!this.state.imageSelected}/>
                    <img src={this.state.file[1]} className={"img-preview"} hidden={!this.state.imageSelected}/>
                      
                    <FormGroup>
                        <Button color="success" onClick={this._predict}
                                disabled={this.state.isLoading}> Predict</Button>
                        <span className="p-1 "/>
                        <Button color="danger" onClick={this._clear}> Clear</Button>
                    </FormGroup>


                    {this.state.isLoading && (
                        <div>
                            <Spinner color="primary" type="grow" style={{width: '5rem', height: '5rem'}}/>

                        </div>
                    )}

                </Form>

                {this.renderPrediction()}


            </div>
        );
    }
}

class CustomNavBar extends React.Component {


    render() {
        const link = APP_CONFIG.code;
        return (
            <Navbar color="light" light fixed expand="md">
                <NavbarBrand href="/">{APP_CONFIG.title}</NavbarBrand>
                <Collapse navbar>
                    <Nav className="ml-auto" navbar>

                    
                    </Nav>
                </Collapse>
            </Navbar>
        )
    }
}


// Create a function to wrap up your component
function App() {
    return (


        <Router>
            <div className="App">
                <CustomNavBar/>
                <div>
                    <main role="main" className="container">
                        <Route exact path="/" component={MainPage}/>
                        <Route exact path="/about" component={About}/>

                    </main>


                </div>
            </div>
        </Router>
    )
}

(async () => {
    const response = await fetch('/config');
    const body = await response.json();

    window.APP_CONFIG = body;

    // Use the ReactDOM.render to show your component on the browser
    ReactDOM.render(
        <App/>,
        rootElement
    )
})();


