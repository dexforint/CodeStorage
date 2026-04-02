### File System Provider Information

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider

Provides information about the file system provider, including whether it's a file system provider and the domains it supports.

````APIDOC
## FileSystemProvider Properties

### Description
Provides access to files and folders on a file system. These APIs work with UXP Manifest version v5 and above.

### isFileSystemProvider
- **Type**: `boolean`
- **Description**: Indicates that this is a `FileSystemProvider`. Useful for type-checking.

### supportedDomains
- **Type**: `Array<Symbol>`
- **Description**: An array of the domains this file system supports. If the file system can open a file picker to the user's `documents` folder, for example, then `userDocuments` will be in this list.

### Example
```javascript
if (fs.supportedDomains.contains(domains.userDocuments)) {
    console.log("We can open a picker to the user's documents.")
}
````

````

--------------------------------

### Headers Iteration Methods

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/Headers

Methods for checking header existence and iterating over headers.

```APIDOC
## has(name)

### Description
Indicates whether the Headers object contains a certain header.

### Method
`has`

### Returns
- `boolean` - True if the Headers object contains the input name, false otherwise.

### Parameters
- **name** (`string`) - Required - The name of the HTTP header.
````

```APIDOC
## forEach(callbackFn, thisArg)

### Description
Executes a callback function once for each key/value pair in the Headers object.

### Method
`forEach`

### Parameters
- **callbackFn** (`function`) - Required - Function to execute for each entry. It takes the following arguments: value, name, this.
- **thisArg** (`Object`) - Optional - Value to use as `this` when executing the callback.
```

```APIDOC
## keys()

### Description
Returns an iterator object allowing iteration through all keys (header names) in the Headers object.

### Method
`keys`

### Returns
- `iterator` - An iterator for the header names.
```

```APIDOC
## values()

### Description
Returns an iterator object allowing iteration through all values (header values) in the Headers object.

### Method
`values`

### Returns
- `iterator` - An iterator for the header values.
```

```APIDOC
## entries()

### Description
Returns an iterator object allowing iteration through all key/value pairs (entries) in the Headers object.

### Method
`entries`

### Returns
- `iterator` - An iterator for the header entries (key-value pairs).
```

---

### Define Panel Entrypoint in UXP Manifest

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-lifecycle-hooks

This JSON snippet illustrates how to declare a panel entrypoint within the `manifest.json` file for a UXP plugin. It specifies the panel's type, a unique ID, label, and size constraints. This configuration is essential for the `entrypoints.setup()` function to correctly associate lifecycle hooks with the defined panel.

```json
{
	// ...
	"entrypoints": [
		{
			"type": "panel",
			"id": "firstPanel",
			"label": "My plugin",
			"minimumSize": { "width": 400, "height": 400 },
			"maximumSize": { "width": 800, "height": 800 },
			"preferredDockedSize": { "width": 400, "height": 400 },
			"preferredFloatingSize": { "width": 600, "height": 600 }
		}
	]
	// ...
}
```

---

### Structure HTML for Multiple UXP Panels

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-panels

This HTML snippet demonstrates how to structure the `index.html` file for a UXP plugin with multiple panels. It includes separate wrapper divs for each panel's content. The second panel's wrapper has an ID for JavaScript referencing, allowing it to be managed programmatically.

```html
<!DOCTYPE html>
<html>
	<head>
		<script src="main.js"></script>
		<link
			rel="stylesheet"
			href="style.css"
		/>
	</head>
	<body>
		<!-- First panel content -->
		<div class="wrapper">
			<sp-heading>First Panel</sp-heading>
			<sp-divider size="L"></sp-divider>
			<sp-body> This is the first panel. </sp-body>
			<sp-button id="open-second-panel">Open Second Panel</sp-button>
		</div>

		<!-- Second panel content -->
		<div
			class="wrapper"
			id="second-panel"
		>
			<sp-heading>Second Panel</sp-heading>
			<sp-divider size="L"></sp-divider>
			<sp-body> This is the second panel. </sp-body>
		</div>
	</body>
</html>
```

---

### Allow-list Domain for Dynamic Image Loading in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/network

Shows how to dynamically load an image using an `<img>` tag in UXP, provided the image's domain is correctly listed in the plugin's `manifest.json` under `requiredPermissions.network.domains`. This ensures the image can be fetched and displayed.

```html
<img
	src="https://picsum.photos/300/200"
	alt="A random image"
/>
```

---

### Addon Definitions

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Defines addon configurations for hybrid plugins, enabling access to C++ native libraries. Note: Premiere Pro currently does not support hybrid plugins.

```APIDOC
## Addon Definition

### Description
Addon definitions for hybrid plugins. A UXP Hybrid plugin is a UXP plugin that can access the power of C++ native libraries. Premiere doesn't support hybrid plugins yet.

### Fields
#### `addon`
*   **Type**: `object`
*   **Default**: `{}`
*   **Description**: An object for configuring addon definitions.
```

---

### Project Item Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/clipprojectitem

This section details various methods available for interacting with project items, such as retrieving their parent project, proxy path, sequence information, and checking their status (e.g., offline, merged clip, multicam clip).

````APIDOC
## getProject

### Description
Get the parent Project of this projectItem.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/projectItem/getProject

### Parameters
#### Query Parameters
- **mediaType** (Constants.MediaType) - Required - Media type can be audio, video or data/caption

### Response
#### Success Response (200)
- **Project** (object) - The parent Project object.

### Response Example
```json
{
  "project": {
    "id": "123",
    "name": "MyProject"
  }
}
````

## getProxyPath

### Description

Returns the proxy path if the project item has a proxy attached.

### Method

GET

### Endpoint

/websites/developer_adobe_premiere-pro_uxp/projectItem/getProxyPath

### Parameters

#### Query Parameters

- **mediaType** (Constants.MediaType) - Required - Media type can be audio, video or data/caption

### Response

#### Success Response (200)

- **string** (string) - The path to the proxy file.

### Response Example

```json
{
	"proxyPath": "/path/to/proxy.mp4"
}
```

## getSequence

### Description

Get the sequence of the Project item.

### Method

GET

### Endpoint

/websites/developer_adobe_premiere-pro_uxp/projectItem/getSequence

### Parameters

#### Query Parameters

- **mediaType** (Constants.MediaType) - Required - Media type can be audio, video or data/caption

### Response

#### Success Response (200)

- **Sequence** (object) - The Sequence object associated with the project item.

### Response Example

```json
{
	"sequence": {
		"id": "456",
		"name": "MySequence"
	}
}
```

## hasProxy

### Description

Indicates whether a proxy has already been attached to the project item.

### Method

GET

### Endpoint

/websites/developer_adobe_premiere-pro_uxp/projectItem/hasProxy

### Parameters

#### Query Parameters

- **mediaType** (Constants.MediaType) - Required - Media type can be audio, video or data/caption

### Response

#### Success Response (200)

- **boolean** (boolean) - True if a proxy is attached, false otherwise.

### Response Example

```json
{
	"hasProxy": true
}
```

## isMergedClip

### Description

Returns true if the clip Project item is a merged clip.

### Method

GET

### Endpoint

/websites/developer_adobe_premiere-pro_uxp/projectItem/isMergedClip

### Parameters

#### Query Parameters

- **mediaType** (Constants.MediaType) - Required - Media type can be audio, video or data/caption

### Response

#### Success Response (200)

- **boolean** (boolean) - True if it is a merged clip, false otherwise.

### Response Example

```json
{
	"isMergedClip": false
}
```

## isMulticamClip

### Description

Returns true if the clip Project item is a multicam clip.

### Method

GET

### Endpoint

/websites/developer_adobe_premiere-pro_uxp/projectItem/isMulticamClip

### Parameters

#### Query Parameters

- **mediaType** (Constants.MediaType) - Required - Media type can be audio, video or data/caption

### Response

#### Success Response (200)

- **boolean** (boolean) - True if it is a multicam clip, false otherwise.

### Response Example

```json
{
	"isMulticamClip": false
}
```

## isOffline

### Description

Returns true if the media is offline.

### Method

GET

### Endpoint

/websites/developer_adobe_premiere-pro_uxp/projectItem/isOffline

### Parameters

#### Query Parameters

- **mediaType** (Constants.MediaType) - Required - Media type can be audio, video or data/caption

### Response

#### Success Response (200)

- **boolean** (boolean) - True if the media is offline, false otherwise.

### Response Example

```json
{
	"isOffline": false
}
```

## isSequence

### Description

Returns true if the project item is a sequence.

### Method

GET

### Endpoint

/websites/developer_adobe_premiere-pro_uxp/projectItem/isSequence

### Parameters

#### Query Parameters

- **mediaType** (Constants.MediaType) - Required - Media type can be audio, video or data/caption

### Response

#### Success Response (200)

- **boolean** (boolean) - True if the project item is a sequence, false otherwise.

### Response Example

```json
{
	"isSequence": true
}
```

## refreshMedia

### Description

Updates representation of the media associated with the project item.

### Method

GET

### Endpoint

/websites/developer_adobe_premiere-pro_uxp/projectItem/refreshMedia

### Parameters

#### Query Parameters

- **mediaType** (Constants.MediaType) - Required - Media type can be audio, video or data/caption

### Response

#### Success Response (200)

- **boolean** (boolean) - Indicates if the media refresh was successful.

### Response Example

```json
{
	"refreshMedia": true
}
```

````

--------------------------------

### Project Properties

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/project

Access and understand the properties of a Premiere Pro project.

```APIDOC
## Project Properties

### Description
Properties of a Premiere Pro project.

### Properties

- **guid** (Guid) - R - The unique identifier of the project.
- **name** (string) - R - The project name.
- **path** (string) - R - The absolute file path to the project file.
````

---

### Using Spectrum Web Component Button

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum

Illustrates how to use an imported Spectrum Web Component (SWC) button. Although the tag name is identical to the Spectrum UXP widget, the underlying implementation differs.

```html
<sp-button variant="primary">I'm a SWC button</sp-button>
```

---

### Enable Alerts in Manifest (JSON)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20DOM/prompt

This JSON configuration snippet shows how to enable alert-related features, including `prompt()`, in the UXP plugin manifest file. This is required for `prompt()` to function in plugins since UXP v7.4.

```json
"featureFlags": {
     "enableAlerts": true
 }
```

---

### Element Selection and Interaction

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHtmlElement

Methods for finding elements within the DOM and simulating user interactions.

````APIDOC
## click()

### Description
Simulates a mouse click on the element.

### Method
POST (conceptually, as it triggers an event)

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// buttonElement.click();
````

### Response

#### Success Response (200)

None (triggers click event)

#### Response Example

None

````

```APIDOC
## getElementsByClassName(name)

### Description
Returns a NodeList of all descendant elements with the specified class name.

### Method
GET

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// const elements = parentElement.getElementsByClassName('my-class');
````

### Response

#### Success Response (200)

- **NodeList** - A live `NodeList` containing all matching descendant elements.

#### Response Example

```json
[
	{
		"tagName": "div",
		"className": "my-class"
	},
	{
		"tagName": "span",
		"className": "my-class"
	}
]
```

````

```APIDOC
## getElementsByTagName(name)

### Description
Returns a NodeList of all descendant elements with the specified tag name.

### Method
GET

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// const paragraphs = document.getElementsByTagName('p');
````

### Response

#### Success Response (200)

- **NodeList** - A live `NodeList` containing all matching descendant elements.

#### Response Example

```json
[
	{
		"tagName": "p"
	},
	{
		"tagName": "p"
	}
]
```

````

```APIDOC
## querySelector(selector)

### Description
Returns the first descendant element that matches the specified CSS selector.

### Method
GET

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// const firstDiv = document.querySelector('div.container');
````

### Response

#### Success Response (200)

- **Element** - The first matching element, or null if no match is found.

#### Response Example

```json
{
	"tagName": "div",
	"className": "container"
}
```

````

```APIDOC
## querySelectorAll(selector)

### Description
Returns a NodeList of all descendant elements that match the specified CSS selector.

### Method
GET

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// const allLinks = document.querySelectorAll('a');
````

### Response

#### Success Response (200)

- **NodeList** - A `NodeList` containing all matching descendant elements.

#### Response Example

```json
[
	{
		"tagName": "a",
		"href": "http://example.com"
	},
	{
		"tagName": "a",
		"href": "http://anothersite.com"
	}
]
```

````

--------------------------------

### CountQueuingStrategy Constructor

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Streams/CountQueuingStrategy

Creates a new CountQueuingStrategy object with a specified high water mark to manage stream backpressure.

```APIDOC
## CountQueuingStrategy(init)

### Description
Creates a new CountQueuingStrategy object with the provided high water mark.

### Method
`new CountQueuingStrategy(init)`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
const strategy = new CountQueuingStrategy({ highWaterMark: 10 });
````

### Response

#### Success Response (200)

Returns a `CountQueuingStrategy` object.

#### Response Example

```json
{
	"strategy": "CountQueuingStrategy"
}
```

````

--------------------------------

### Manifest Settings for Local WebView Rendering

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

Specifies the necessary manifest.json configurations to enable local content rendering and message bridging for UXP WebViews. Ensures proper communication between the plugin and the WebView.

```json
"requiredPermissions": {
 "webview": {
     "allow": "yes",
     "domains": [],
     "allowLocalRendering": "yes",
     "enableMessageBridge": "localOnly"
  }
}
````

---

### Event Handling API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLOptionElement

APIs for adding, removing, and dispatching events on elements.

````APIDOC
## addEventListener(eventName, callback, options)

### Description
Attaches an event handler to the element.

### Method
POST (Implied)

### Endpoint
EventTarget.addEventListener

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **eventName** (*) - The name of the event to listen for.
- **callback** (*) - The function to call when the event is triggered.
- **options** (boolean | Object) - A boolean value denoting capture value or options object. Currently supports only capture in options object ({ capture: bool_value }).

### Request Example
```json
{
  "eventName": "click",
  "callback": "function(event) { console.log('Clicked!'); }",
  "options": {
    "capture": false
  }
}
````

### Response

#### Success Response (200)

No specific return value documented, operation is side-effect based.

#### Response Example

```json
{}
```

## removeEventListener(eventName, callback, options)

### Description

Removes an event handler from the element.

### Method

DELETE (Implied)

### Endpoint

EventTarget.removeEventListener

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

- **eventName** (\*) - The name of the event.
- **callback** (\*) - The event handler function to remove.
- **options** (boolean | Object) - A boolean value denoting capture value or options object. Currently supports only capture in options object ({ capture: bool_value }).

### Request Example

```json
{
	"eventName": "click",
	"callback": "function(event) { console.log('Clicked!'); }",
	"options": {
		"capture": false
	}
}
```

### Response

#### Success Response (200)

No specific return value documented, operation is side-effect based.

#### Response Example

```json
{}
```

## dispatchEvent(event)

### Description

Dispatches an event into the event model, an event that you created (or that was created in an event object) is dispatched on the element.

### Method

POST (Implied)

### Endpoint

EventTarget.dispatchEvent

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

- **event** (\*) - The Event object to dispatch.

### Request Example

```json
{
	"event": "new Event('customEvent')"
}
```

### Response

#### Success Response (200)

- **eventDispatched** (boolean) - True if the event was dispatched successfully, false otherwise.

#### Response Example

```json
{
	"eventDispatched": true
}
```

````

--------------------------------

### Enable SWC Support in manifest.json

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/user-interfaces

Configuration snippet for the `manifest.json` file to enable Spectrum Web Components (SWC) support within a UXP plugin. This flag is necessary for SWC functionality to work correctly.

```json
{
  "featureFlags": {
    "enableSWCSupport": true
  }
}
````

---

### Access Premiere Pro DOM via UXP

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference

This code snippet imports the Premiere Pro UXP module, granting access to the Premiere DOM. This is the entry point for interacting with Premiere Pro functionalities such as opening documents, modifying them, and executing menu items.

```javascript
const app = require("premierepro");
```

---

### Edit Plugin UI with HTML

Source: https://developer.adobe.com/premiere-pro/uxp/plugins

This HTML code defines the user interface for the UXP plugin. It includes a heading, a container for displaying plugin information, and buttons to trigger actions. It links to external CSS and JavaScript files for styling and functionality.

```html
<!DOCTYPE html>
<html>
	<head>
		<script src="main.js"></script>
		<link
			rel="stylesheet"
			href="style.css"
		/>
	</head>
	<body>
		<h4>Application Info</h4>
		<div class="main-div">
			<sp-body id="plugin-body"> </sp-body>
		</div>
		<footer>
			<sp-button id="btnPopulate">Populate Application Info</sp-button>
			<sp-button id="clear-btn">Clear Application Info</sp-button>
		</footer>
	</body>
</html>
```

---

### Load and display external HTML in UXP (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-modal-dialogs

This JavaScript code snippet demonstrates how to load HTML content from an external file (`_dialog.html`) using the `fetch()` method in UXP. It handles the response as text and uses the loaded HTML to populate a dialog's `innerHTML`. It also shows how to dynamically update placeholders in the HTML with data from the manifest and host environment.

```javascript
const { entrypoints, host, versions } = require("uxp");
const manifest = require("./manifest.json");
const os = require("os");

entrypoints.setup({
	commands: {
		"about-command": async () => {
			// Load the HTML content from an external file
			// In this case, the _dialog.html file is located in the
			// same directory as the main.js file
			const dialogHtml = await fetch("./_dialog.html") // 👆
				.then((res) => res.text()); // 👆 handle the response as text
			console.log("About command");
			const dialog = document.createElement("dialog");
			dialog.innerHTML = dialogHtml.trim();
			// ...
			// Replace the placeholders with the actual values. It is the
			// equivalent of using Template Literals in the previous example
			dialog.querySelector("#version").textContent = manifest.version;
			dialog.querySelector("#app-name").textContent = host.name;
			dialog.querySelector("#app-version").textContent = host.version;
			dialog.querySelector("#platform").textContent = os.platform();
			dialog.querySelector("#uxp-version").textContent = versions.uxp;
			dialog.querySelector("#plugin-version").textContent = versions.plugin;
			// ...
			document.body.appendChild(dialog);
			// ...
			const result = await dialog.uxpShowModal({
				/* ... */
			});
		},
	},
});
```

---

### Shadow DOM

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLSelectElement

Methods for attaching and managing Shadow DOM.

```APIDOC
## attachShadow(init)

### Description
Attaches a shadow DOM tree to the specified element and returns a reference to its ShadowRoot. This feature is behind a feature flag and requires `enableSWCSupport` to be enabled in the plugin manifest.

### Method
`attachShadow`

### Parameters
#### Path Parameters
- **init** - An object which contains the fields : mode(open/closed) , delegatesFocus ,slotAssignment

### Returns
`ShadowRoot`

### See
Element - attachShadow
```

---

### Entrypoint Definition

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Defines an entrypoint for the plugin, specifying its type, ID, label, and other visual or sizing properties.

```APIDOC
## EntrypointDefinition

### Description
Represents an entrypoint that the plugin provides. An entrypoint is a point of entry for the plugin. It is used to identify the code that implements the entrypoint.

### Properties
#### Required Properties
*   **`type`** (string) - Required - The type of entrypoint. Currently, only Commands and Panels are supported (`"command"` or `"panel"`).
*   **`id`** (string) - Required - A unique identifier for the entrypoint.
*   **`label`** (string) - Required - The user-facing label for the entrypoint.

#### Optional Properties
*   **`description`** (string) - Optional - A description of the entrypoint.
*   **`shortcut`** (object) - Optional - Configuration for keyboard shortcuts associated with the entrypoint.
*   **`icon`** (IconDefinition) - Optional - An `IconDefinition` object for the entrypoint's icon.
*   **`minimumSize`** (object) - Optional - The minimum size constraints for the entrypoint's UI.
*   **`maximumSize`** (object) - Optional - The maximum size constraints for the entrypoint's UI.
*   **`preferredDockedSize`** (object) - Optional - The preferred size when the entrypoint is docked.
*   **`preferredFloatingSize`** (object) - Optional - The preferred size when the entrypoint is floating.
```

---

### Declare UXP Plugin Entrypoints in Manifest

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/entrypoints

This JSON snippet shows how to declare 'command' and 'panel' entrypoints within the `manifest.json` file for a UXP plugin in Premiere Pro. It specifies the type, ID, and label for each entrypoint.

```json
{
	"manifestVersion": 5,
	"id": "test1234",
	"name": "Test Plugin",
	"version": "1.0.0",
	"host": { "app": "premierepro", "minVersion": "25.6.0" },
	"main": "index.html",
	"entrypoints": [
		{
			"type": "command",
			"id": "firstCommand",
			"label": { "default": "Run a Function" }
		},
		{
			"type": "panel",
			"id": "firstPanel",
			"label": { "default": "First Panel" }
		}
	]
}
```

---

### Caption Track Instance Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/captiontrack

Methods for interacting with a caption track instance.

```APIDOC
## CaptionTrack Instance Methods

### getIndex

#### Description
Gets the index of the track within the track group.

#### Method
GET

#### Endpoint
`/captiontrack/getIndex`

#### Return Value
- **(number)** - The index representing the track index of this track within the track group. Minimum Version: 25.0

### getMediaType

#### Description
Gets the UUID representing the underlying media type of the track.

#### Method
GET

#### Endpoint
`/captiontrack/getMediaType`

#### Return Value
- **(Guid)** - UUID representing the underlying media type of this track. Minimum Version: 25.0

### getTrackItems

#### Description
Returns the track items of the specified media type from the given track.

#### Method
GET

#### Endpoint
`/captiontrack/getTrackItems`

#### Query Parameters
- **trackItemType** (number) - Description: Specifies the type of track items to retrieve.
- **includeEmptyTrackItems** (boolean) - Description: Whether to include empty track items.

#### Return Value
- **(\[])** - An array of track items. Minimum Version: 25.0

### isMuted

#### Description
Gets the mute state of the track.

#### Method
GET

#### Endpoint
`/captiontrack/isMuted`

#### Return Value
- **(boolean)** - True if the track is muted, false otherwise. Minimum Version: 25.0

### setMute

#### Description
Sets the mute state of the track to muted or unmuted.

#### Method
POST

#### Endpoint
`/captiontrack/setMute`

#### Query Parameters
- **mute** (boolean) - Description: Set to true to mute the track, false to unmute.
```

---

### HTML Content for the Second Panel

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-panels

Provides the HTML content for the second panel of a UXP plugin. This HTML fragment is intended to be fetched and injected into the DOM of the main `index.html` file.

```html
<!-- This is what used to be inside the second panel's <div> wrapper -->
<sp-heading>Second Panel</sp-heading>
<sp-divider size="L"></sp-divider>
<sp-body> This is the second panel. </sp-body>
```

---

### Check Supported File System Domains (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider

Demonstrates how to check if a file system provider supports accessing user documents using the `supportedDomains` property and `domains.userDocuments`. This is useful for ensuring compatibility before attempting to open a file picker.

```javascript
if (fs.supportedDomains.contains(domains.userDocuments)) {
	console.log("We can open a picker to the user's documents.");
}
```

---

### Icon Configuration with Theming (JSON)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Specifies icons for different UI themes (e.g., light, dark). This allows the plugin to display icons that are visually appropriate for the user's current theme setting in the application.

```json
{
    "path": "icon-light.png",
    "width": 24,
    "height": 24,
    "theme": ["lightest", "light"]
},
{
    "path": "icon-dark.png",
    "width": 24,
    "height": 24,
    "theme": ["darkest", "dark"]
}
```

---

### Basic HTML Elements for UXP UI

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/user-interfaces

Demonstrates the use of standard HTML elements for creating a user interface in UXP plugins. This approach offers maximum control over styling via CSS but requires custom implementation for Adobe's design system adherence.

```html
<div class="container">
	<button class="primary-button">Click me</button>
	<input
		type="text"
		placeholder="Enter text"
	/>
</div>
```

---

### Define Panel Lifecycle Hooks in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/entrypoints

Defines panel lifecycle hooks (`create`, `show`, `hide`, `destroy`) within the `panels` property of `entrypoints.setup()`. These hooks are associated with panel entrypoints declared in `manifest.json`. The `rootNode` parameter allows programmatic manipulation of the DOM. Note: `hide()` and `destroy()` hooks are currently not working as expected in Premiere.

```javascript
const { entrypoints } = require("uxp");
entrypoints.setup({
	panels: {
		firstPanel: {
			// 👈 must match the id of the entrypoint
			//    of type "panel" from manifest.json
			create(rootNode) {
				console.log("Panel created");
			},
			show(rootNode) {
				console.log("Panel shown");
			},
			hide(rootNode) {
				console.log("Panel hidden");
			},
			destroy(rootNode) {
				console.log("Panel destroyed");
			},
		},
		secondPanel: {
			/*...*/
		}, // 👈  add properties for each additional
		//    "panel" entrypoint in manifest.json
	},
});
```

---

### Premiere Pro UXP Plugin Manifest (JSON)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-modal-dialogs

Defines the manifest file for a Premiere Pro UXP plugin. It specifies plugin details, host application compatibility, required permissions, feature flags, and entry points for panels. This includes icon definitions for different themes and scales.

```json
{
  "id": "Test-modaldialog",
  "name": "Test-modaldialog",
  "shortname": "3pstarterplugin",
  "version": "1.0.0",
  "main": "index.html",
  "host": { "app": "premierepro", "minVersion": "25.6.0" },
  "manifestVersion": 5,
  "requiredPermissions": { "localFileSystem": "request" },
  "featureFlags": { "enableAlerts": true },
  "entrypoints": [
    {
      "id": "starterpanel",
      "type": "panel",
      "minimumSize": { "width": 430, "height": 500 },
      "maximumSize": { "width": 2000, "height": 2000 },
      "preferredDockedSize": { "width": 230, "height": 300 },
      "preferredFloatingSize": { "width": 400, "height": 300 },
      "label": { "default": "PremierePro Modal Dialog" },
      "icons": [
        {
          "width": 23, "height": 23, "path": "icons/dark.png",
          "scale": [ 1, 2 ], "theme": [ "darkest", "dark", "medium" ]
        },
        {
          "width": 23, "height": 23, "path": "icons/light.png",
          "scale": [ 1, 2 ], "theme": [ "lightest", "light" ]
        },
      ]
    }
  ],
  "icons": [
    {
      "width": 48, "height": 48, "path": "icons/plugin-icon.png",
      "theme": [

```

---

### Query and Select Elements (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20DOM/Element

Demonstrates JavaScript methods for selecting elements within the DOM using various criteria, including class names, tag names, CSS selectors, and hierarchical relationships. These are standard DOM APIs.

```javascript
// Get elements by class name
const elementsByClass = document.getElementsByClassName("className");

// Get elements by tag name
const elementsByTag = document.getElementsByTagName("tagName");

// Query selector for the first matching element
const firstElement = document.querySelector("cssSelector");

// Query selector all for all matching elements
const allElements = document.querySelectorAll("cssSelector");

// Get bounding client rectangle
const rect = element.getBoundingClientRect();

// Find closest element matching a selector
const closestElement = element.closest("selectorString");

// Check if element matches a selector
const matchesSelector = element.matches("selectorString");
```

---

### Render Progress Bar - HTML (UXP)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-progressbar

Demonstrates how to render a basic progress bar using the sp-progressbar component in UXP. It showcases setting the maximum and current values.

```html
<sp-progressbar
	max="100"
	value="50"
></sp-progressbar>
```

---

### Configure WebView Permissions in UXP Manifest

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

This JSON configuration shows how to set up the necessary permissions for the WebView component within your UXP plugin's `manifest.json` file. It includes enabling the 'webview' permission, specifying allowed domains, and configuring the message bridge for inter-process communication.

```json
// `manifest.json`
{
	"manifestVersion": 5,
	"requiredPermissions": {
		"webview": {
			"allow": "yes",
			// domains --> string[] | "all"
			"domains": ["https://*.adobe.com", "https://*.google.com"],
			// enableMessageBridge can use either of these data "localAndRemote" | "localOnly" | "no"
			"enableMessageBridge": "localAndRemote"
		}
	}
}
```

---

### Shadow DOM

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLButtonElement

Methods for attaching and managing Shadow DOM trees.

````APIDOC
## attachShadow(init)

### Description
Attaches a shadow DOM tree to the specified element and returns a reference to its ShadowRoot. This feature is behind a feature flag (`enableSWCSupport`).

### Method

[Implicitly defined by function signature]

### Endpoint

[Not applicable for this method]

### Parameters
#### Path Parameters

[None]

#### Query Parameters

[None]

#### Request Body

[None]

### Request Example

```javascript
// Requires enabling 'enableSWCSupport' in plugin manifest.
// const shadowRoot = element.attachShadow({ mode: 'open' });
````

### Response

#### Success Response (200)

- **ShadowRoot** (`ShadowRoot`) - A reference to the created ShadowRoot.

#### Response Example

[Conceptual example, actual object structure depends on ShadowRoot implementation]

```json
{
	"//": "Represents a ShadowRoot object"
}
```

````

--------------------------------

### WritableStreamDefaultController API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Streams/WritableStreamDefaultController

This section details the methods available on the WritableStreamDefaultController, including how to abort pending operations and how to signal errors.

```APIDOC
## WritableStreamDefaultController API

### Description
Provides control over a WritableStream, allowing for abortion of pending operations and forceful error signaling.

### Methods

#### `signal`

##### Description
Returns an `AbortSignal` that can be used to abort the pending write or close operation when the stream is aborted.

##### Method
GET

##### Endpoint
`/websites/developer_adobe_premiere-pro_uxp#window.WritableStreamDefaultController.signal`

##### Parameters

###### Query Parameters
- **signal** (AbortSignal) - N/A - An AbortSignal object.

##### Response

###### Success Response (200)
- **AbortSignal** (object) - The signal object for aborting operations.

###### Response Example
```json
{
  "signal": "AbortSignal Object"
}
````

#### `error(message)`

##### Description

Closes the controlled writable stream, making all future interactions with it fail with the given error message.

##### Method

POST

##### Endpoint

`/websites/developer_adobe_premiere-pro_uxp#window.WritableStreamDefaultController.error`

##### Parameters

###### Path Parameters

- **message** (string) - Required - The error message to be used when closing the stream.

##### Request Example

```json
{
	"message": "An error occurred during stream operation."
}
```

##### Response

###### Success Response (200)

- **status** (string) - Indicates the success of the operation.

###### Response Example

```json
{
	"status": "Stream closed with error."
}
```

````

--------------------------------

### Programmatically Open UXP Panel via Plugin Manager

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-panels

This code snippet demonstrates how to find the current plugin within the `pluginManager` and then programmatically open a specific panel using its entrypoint ID. This is useful for controlling panel visibility through user interactions or other plugin logic.

```javascript
const me = [...pluginManager.plugins].find(
  (plugin) => plugin.id === PLUGIN_ID
);
me?.showPanel("uxp-second-panel"); // Opens the panel by entrypoint ID

````

---

### Node Tree Traversal API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLOptionElement

APIs for checking node containment and retrieving the root node.

````APIDOC
## contains(node)

### Description
Returns a boolean indicating whether a specified node is a descendant of the element.

### Method
GET (Implied)

### Endpoint
Node.contains

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **node** (Node) - The node to check for.

### Request Example
```json
{
  "node": "<p>Descendant</p>"
}
````

### Response

#### Success Response (200)

- **containsNode** (boolean) - True if the node is a descendant, false otherwise.

#### Response Example

```json
{
	"containsNode": true
}
```

## getRootNode(options)

### Description

Returns the root node of the current node.

### Method

GET (Implied)

### Endpoint

Node.getRootNode

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

- **options** (Object) - An options object. Currently supports `composed` which, if true, indicates that the root node should be traversed up to the document root, irrespective of shadow DOM boundaries.

### Request Example

```json
{
	"options": {
		"composed": true
	}
}
```

### Response

#### Success Response (200)

- **rootNode** (Node) - The root node.

#### Response Example

```json
{
	"rootNode": "#document"
}
```

````

--------------------------------

### Element Query APIs

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLDialogElement

APIs for selecting elements within the DOM.

```APIDOC
## querySelector(selector)

### Description
Selects the first element that is a descendant of this element which matches the specified group of selectors.

### Method
GET

### Endpoint
Not Applicable (this is a JavaScript method)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
const element = parentElement.querySelector('.my-class');
````

### Response

#### Success Response (200)

- **Element** (Element) - The first matching element, or null if no matches are found.

#### Response Example

```json
{
	"element": "<div class=\"my-class\">...</div>"
}
```

## querySelectorAll(selector)

### Description

Returns a static (nondynamic) `NodeList` representing a list of the element's descendants that match the specified group of selectors.

### Method

GET

### Endpoint

Not Applicable (this is a JavaScript method)

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

None

### Request Example

```javascript
const elements = parentElement.querySelectorAll("p");
```

### Response

#### Success Response (200)

- **NodeList** (NodeList) - A `NodeList` containing all matching elements.

#### Response Example

```json
{
	"elements": ["<p>Paragraph 1</p>", "<p>Paragraph 2</p>"]
}
```

````

--------------------------------

### Create a Themed Heading with Spectrum UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-html/Hierarchy/h1

Shows how to use the Spectrum UXP `sp-heading` component to create a theme-aware heading. This component integrates with Spectrum's design system for consistent styling across Adobe applications. It requires the Spectrum UXP library.

```html
<sp-heading size="L">Hello, World</sp-heading>
````

---

### Root Node Access and Event Handling

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLVideoElement

Provides methods for accessing the root node of the UXP tree and for managing event listeners on elements.

````APIDOC
## getRootNode(options)

### Description
Retrieves the root node of the UXP document.

### Method
*getRootNode*

### Endpoint
N/A (Method within a class)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **options** (Object) - Optional configuration for retrieving the root node.

### Request Example
```javascript
const root = uxp.getRootNode();
````

### Response

#### Success Response (200)

- **Node** - The root node of the UXP document.

#### Response Example

```json
{
	"nodeType": "root",
	"id": "1"
}
```

## addEventListener(eventName, callback, options)

### Description

Attaches an event listener to an element. See EventTarget - addEventListener for more details.

### Method

_addEventListener_

### Endpoint

N/A (Method within a class)

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

- **eventName** (\*) - The name of the event to listen for.
- **callback** (\*) - The function to call when the event is triggered.
- **options** (boolean | Object) - Boolean value denoting capture value or options object. Currently supports only capture in options object ({ capture: bool_value }).

### Request Example

```javascript
myElement.addEventListener("click", handleClick, { capture: true });
```

### Response

#### Success Response (200)

None (This is a method call).

#### Response Example

None

## removeEventListener(eventName, callback, options)

### Description

Removes an event listener from an element. See EventTarget - removeEventListener for more details.

### Method

_removeEventListener_

### Endpoint

N/A (Method within a class)

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

- **eventName** (\*) - The name of the event to remove the listener for.
- **callback** (\*) - The callback function that was originally added.
- **options** (boolean | Object) - Boolean value denoting capture value or options object. Currently supports only capture in options object ({ capture: bool_value }).

### Request Example

```javascript
myElement.removeEventListener("click", handleClick, { capture: true });
```

### Response

#### Success Response (200)

None (This is a method call).

#### Response Example

None

## dispatchEvent(event)

### Description

Dispatches an event on the element.

### Method

_dispatchEvent_

### Endpoint

N/A (Method within a class)

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

- **event** (\*) - The event object to dispatch.

### Request Example

```javascript
const myEvent = new Event("customEvent");
myElement.dispatchEvent(myEvent);
```

### Response

#### Success Response (200)

- **boolean** - True if the event was dispatched successfully, false otherwise.

#### Response Example

```json
true
```

````

--------------------------------

### Element Interaction and Querying

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHeadElement

Methods for simulating clicks and querying elements by class name, tag name, or CSS selectors.

```APIDOC
## click()

### Description
Simulates a mouse click on the element.

### Method
N/A (JavaScript method)

### Endpoint
N/A

### Parameters
N/A

### Request Example
```javascript
// element.click();
````

### Response

N/A

## getElementsByClassName(name)

### Description

Returns a live `NodeList` collection of all elements in the document or a specific element that have the specified class name.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```javascript
// const elementsWithClass = document.getElementsByClassName('my-class');
```

### Response

#### Success Response (200)

Returns a `NodeList` of elements.

#### Response Example

N/A

## getElementsByTagName(name)

### Description

Returns a live `NodeList` collection of all elements in the document or a specific element that have the specified tag name.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```javascript
// const divElements = document.getElementsByTagName('div');
```

### Response

#### Success Response (200)

Returns a `NodeList` of elements.

#### Response Example

N/A

## querySelector(selector)

### Description

Returns the first `Element` within the document (or within the element's subtree) that matches the specified group of selectors.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```javascript
// const firstParagraph = document.querySelector('p');
```

### Response

#### Success Response (200)

Returns the first matching `Element`, or `null` if no match is found.

#### Response Example

N/A

## querySelectorAll(selector)

### Description

Returns a non-live `NodeList` representing a list of the document's elements that match the specified group of selectors.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```javascript
// const allListItems = document.querySelectorAll('ul li');
```

### Response

#### Success Response (200)

Returns a `NodeList` of matching elements.

#### Response Example

N/A

````

--------------------------------

### Read File Asynchronously - UXP FSAPI

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/fs

Reads data from a specified file path asynchronously. Supports binary or encoded text formats via options. Returns a Promise with the file content. If no callback is provided, a Promise is returned.

```javascript
const data = await fs.readFile("plugin-data:/binaryFile.obj");
````

```javascript
const text = await fs.readFile("plugin-data:/textFile.txt", { encoding: "utf-8" });
```

---

### Open External URL with Shell API (UXP)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/shell/Shell

Opens a given URL in the system's default application for the URL's scheme. The 'file' scheme is not permitted; use `openPath` for local files. Returns a promise resolving with an empty string on success or an error message on failure. Requires UXP Manifest v5.

```javascript
const { shell } = require("uxp");

shell.openExternal("https://www.adobe.com/");
shell.openExternal("https://www.adobe.com/", "User consent message");

// Example with specific map applications
shell.openExternal("maps://?address=345+Park+Ave+San+Jose"); // for MacOS
shell.openExternal("bingmaps:?q=345+Park+Ave+San+Jose,+95110"); // for Windows
```

---

### Focus and Blur API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLDialogElement

APIs for managing element focus.

```APIDOC
## focus()

### Description
Sets focus to the element.

### Method
`focus`

### Parameters
None

---

## blur()

### Description
Removes focus from the element.

### Method
`blur`

### Parameters
None
```

---

### Render an sp-link Component

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-link

This snippet demonstrates how to render a basic sp-link component, which when clicked, launches a webpage in the user's default browser. It requires the `href` attribute to specify the URL. If `href` is omitted, no browser will launch.

```html
<sp-link href="https://adobe.com">Adobe</sp-link>
```

---

### Element Event Handling

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLSelectElement

Methods for simulating clicks and capturing pointer events.

```APIDOC
## click()

### Description
Simulates a mouse click on the element.

### Method
`click`
```

````APIDOC
## setPointerCapture(pointerId)

### Description
Sets pointer capture for the element. This implementation does not dispatch the `gotpointercapture` event on the element. Throws `DOMException` if the element is not connected to the DOM.

### Method
`setPointerCapture`

### Parameters
#### Path Parameters
- **pointerId** (number) - The unique identifier of the pointer to be captured.

### Throws
- `DOMException` If the element is not connected to the DOM.

### See
Element - setPointerCapture

### Example
```javascript
function beginSliding(e) {
     slider.setPointerCapture(e.pointerId);
     slider.addEventListener("pointermove", slide);
 }

 function stopSliding(e) {
     slider.releasePointerCapture(e.pointerId);
     slider.removeEventListener("pointermove", slide);
 }

 function slide(e) {
     slider.style.left = e.clientX;
 }

 const slider = document.getElementById("slider");

 slider.addEventListener("pointerdown", beginSliding);
 slider.addEventListener("pointerup", stopSliding);
````

````

--------------------------------

### Pointer Capture API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLOptionElement

APIs related to checking and managing pointer capture on elements.

```APIDOC
## hasPointerCapture(pointerId)

### Description
Checks if the element has pointer capture for the specified pointer.

### Method
GET (Implied)

### Endpoint
Element.hasPointerCapture

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
{
  "pointerId": 12345
}
````

### Response

#### Success Response (200)

- **pointerCaptureStatus** (boolean) - True if the element has pointer capture for the specified pointer, false otherwise.

#### Response Example

```json
{
	"pointerCaptureStatus": true
}
```

````

--------------------------------

### Shadow DOM and Focus Management

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHtmlElement

Methods for managing Shadow DOM and element focus.

```APIDOC
## attachShadow(init)

### Description
Attaches a shadow DOM tree to the specified element and returns a reference to its ShadowRoot. This feature requires `enableSWCSupport` to be enabled in the plugin manifest.

### Method
POST (conceptually, as it creates a ShadowRoot)

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **init** (object) - Required - An object which contains the fields: mode (open/closed), delegatesFocus, slotAssignment.

### Request Example
```javascript
// Example usage
// const shadowRoot = element.attachShadow({ mode: 'open' });
````

### Response

#### Success Response (200)

- **ShadowRoot** (object) - A reference to the created ShadowRoot.

#### Response Example

```json
{
	"type": "ShadowRoot"
}
```

````

```APIDOC
## focus()

### Description
Attempts to set focus to the element.

### Method
POST (conceptually, as it changes focus state)

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// element.focus();
````

### Response

#### Success Response (200)

None (modifies focus state)

#### Response Example

None

````

```APIDOC
## blur()

### Description
Removes focus from the element.

### Method
POST (conceptually, as it changes focus state)

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// element.blur();
````

### Response

#### Success Response (200)

None (modifies focus state)

#### Response Example

None

````

--------------------------------

### Invoke Responder Plugin Commands and Panels

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/inter-plugin-comm

This JavaScript code shows how to interact with a found Responder plugin. It demonstrates invoking a command using `invokeCommand()` and showing a panel using `showPanel()`, passing the respective entrypoint IDs.

```javascript
responderPlugin.invokeCommand("simpleCommand");
responderPlugin.showPanel("simplePanel");
````

---

### Pointer Capture APIs

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLDialogElement

APIs for managing pointer capture on an element.

````APIDOC
## setPointerCapture(pointerId)

### Description
Sets pointer capture for the element. This implementation does not dispatch the `gotpointercapture` event on the element.

### Method
POST

### Endpoint
Not Applicable (this is a JavaScript method)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
function beginSliding(e) {
     slider.setPointerCapture(e.pointerId);
     slider.addEventListener("pointermove", slide);
 }
````

### Response

#### Success Response (200)

No specific return value, but pointer capture is established.

#### Response Example

```json
{
	"status": "success"
}
```

### Throws

- **DOMException** - If the element is not connected to the DOM.

### See

- Element - setPointerCapture

### Since

- v7.1

## releasePointerCapture(pointerId)

### Description

Releases pointer capture for the element. This implementation does not dispatch the `lostpointercapture` event on the element.

### Method

DELETE

### Endpoint

Not Applicable (this is a JavaScript method)

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

None

### Request Example

```javascript
function stopSliding(e) {
	slider.releasePointerCapture(e.pointerId);
	slider.removeEventListener("pointermove", slide);
}
```

### Response

#### Success Response (200)

No specific return value, but pointer capture is released.

#### Response Example

```json
{
	"status": "success"
}
```

### See

- Element - releasePointerCapture

### Since

- v7.1

## hasPointerCapture(pointerId)

### Description

Checks if the element has pointer capture for the specified pointer.

### Method

GET

### Endpoint

Not Applicable (this is a JavaScript method)

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

None

### Request Example

```javascript
if (element.hasPointerCapture(e.pointerId)) {
	console.log("Pointer capture is active");
}
```

### Response

#### Success Response (200)

- **boolean** (boolean) - True if the element has pointer capture for the specified pointer, false otherwise.

#### Response Example

```json
{
	"hasCapture": true
}
```

### See

- Element - hasPointerCapture

### Since

- v7.1

````

--------------------------------

### Configure Webview Permissions

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Allows plugins to use webviews for displaying web content or complex UI. Configure allowed domains and message bridge enablement. The 'allow' property must be 'yes' to enable webviews.

```json
{
  "allow": "yes",
  "domains": ["https://example.com"],
  "enableMessageBridge": "localAndRemote"
}
````

---

### Create XMPDateTime Objects in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPDateTime

Demonstrates how to instantiate the XMPDateTime class using various methods. It covers creation from a JavaScript Date object, an ISO 8601 formatted string, and with no arguments for the current date and time. The XMPDateTime class is imported from the 'uxp'.xmp module.

```javascript
const { XMPDateTime } = require("uxp").xmp;

// 1. Creating using Date object
const xdt1 = new XMPDateTime(new Date());

// 2. Creating using iso8601Date
const xdt2 = new XMPDateTime("2007-04-10T17:54:50+01:00");

// 3. Creating with no arguments
const xdt3 = new XMPDateTime();
```

---

### Copy File Asynchronously (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/fs

Copies a file from a source path to a destination path asynchronously. Returns a Promise that resolves with 0 on success or throws an error. Accepts source and destination paths, optional flags, and an optional callback.

```javascript
const data = fs.copyFile("plugin-data:/copyFrom.txt", "plugin-temp:/copyTo.txt");
```

---

### Loading Local HTML Content in UXP WebView

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

Demonstrates how to load local HTML files from plugin, plugin-data, and plugin-temp folders into a UXP WebView using specific protocol URLs. This feature is available from UXP v8.0.0 onwards.

```html
<webview src="plugin:/webview.html"></webview>
<webview src="plugin-data:/webview-in-plugin-data.html"></webview>
<webview src="plugin-temp:/webview-in-plugin-temp.html"></webview>
```

---

### TransformStreamDefaultController API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Streams/TransformStreamDefaultController

API reference for the TransformStreamDefaultController, detailing its properties and methods for managing transform streams.

```APIDOC
## TransformStreamDefaultController API

### Description
Provides an interface for controlling a `TransformStream`.

### Methods

#### `desiredSize` Property

*   **Description**: Returns the desired size to fill the readable side's internal queue. It can be negative if the queue is over-full.
*   **Type**: `number`

#### `enqueue(chunk)` Method

*   **Description**: Enqueues the given chunk in the readable side of the controlled transform stream.
*   **Throws**: `TypeError` if the stream is not readable.
*   **Parameters**:
    *   `chunk` (`*`) - Required - The chunk being queued. A chunk is a single piece of data.

#### `error(reason)` Method

*   **Description**: Errors both the readable side and the writable side of the controlled transform stream, making all future interactions fail with the given error. Any queued chunks will be discarded.
*   **Parameters**:
    *   `reason` (`string`) - Required - The error reason.

#### `terminate()` Method

*   **Description**: Closes the readable side and errors the writable side of the controlled transform stream. Useful when the transformer only needs to consume a portion of the chunks written to the writable side.

### Related

*   [TransformStream]
*   [WritableStream]

```

---

### Querying Elements with CSS Selectors (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLinkElement

Provides methods to select elements using CSS selectors. querySelector returns the first matching element, while querySelectorAll returns all matching elements.

```javascript
/**
 * Returns the first element in the document that matches the specified selector.
 * @param {string} selector - The CSS selector to match.
 * @returns {Element|null} The first matching element, or null if no match is found.
 */
function querySelector(selector) {
	// Implementation details...
}

/**
 * Returns a NodeList of all elements in the document that match the specified selector.
 * @param {string} selector - The CSS selector to match.
 * @returns {NodeList} A NodeList containing all matching elements.
 */
function querySelectorAll(selector) {
	// Implementation details...
}
```

---

### Manage Element Focus and State (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20DOM/Element

Covers basic element interaction methods such as focusing and blurring elements, as well as triggering a click event. These are standard browser DOM APIs.

```javascript
// Focus an element
element.focus();

// Blur an element
element.blur();

// Click an element
element.click();
```

---

### Configure Responder Plugin Manifest with Entrypoints

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/inter-plugin-comm

This JSON configuration for the Responder plugin's `manifest.json` exposes two entrypoints: a panel with `id: "simplePanel"` and a command with `id: "simpleCommand"`. These are the points that the Requester plugin can invoke.

```json
{
	// ...
	"entrypoints": [
		{
			"id": "simplePanel",
			"type": "panel",
			"label": { "default": "Main Panel" }
			// ...
		},
		{
			"id": "simpleCommand",
			"type": "command",
			"label": { "default": "Simple Command" }
		},
		{
			"id": "commandWithInput",
			"type": "command",
			"label": { "default": "Command With Input" }
		}
	]
	// ...
}
```

---

### Trigger or Simulate Clicks and Focus/Blur (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLMenuElement

Methods like click, focus, and blur allow programmatic interaction with UI elements. click simulates a mouse click on the element, while focus and blur manage the element's focus state. These are essential for creating interactive user interfaces and automating user actions.

```javascript
/**
 * Simulates a mouse click on the element.
 */
Element.prototype.click = function () {};

/**
 * Moves the focus to the element.
 */
Element.prototype.focus = function () {};

/**
 * Removes focus from the element.
 */
Element.prototype.blur = function () {};
```

---

### Query Elements Using CSS Selectors (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLMenuElement

querySelector and querySelectorAll allow developers to select elements using CSS selectors. querySelector returns the first element that matches the selector, while querySelectorAll returns all matching elements as a NodeList. These are powerful tools for precise DOM selection.

```javascript
/**
 * Returns the first element in the document that matches the specified selector.
 * @param {string} selector - The CSS selector to match.
 * @returns {Element|null} The first matching element, or null if no match is found.
 */
Element.prototype.querySelector = function (selector) {};

/**
 * Returns a NodeList of all elements in the document that match the specified selector.
 * @param {string} selector - The CSS selector to match.
 * @returns {NodeList} A NodeList containing all matching elements.
 */
Element.prototype.querySelectorAll = function (selector) {};
```

---

### Basic sp-textarea Component

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-textarea

Renders a standard text area with an associated label. It takes a placeholder for user input and a slot for the label element. This is the foundational component for text input.

```html
<sp-textarea placeholder="Enter your name">
	<sp-label slot="label">Name</sp-label>
</sp-textarea>
```

---

### Plugin Localization with StringsDefinition (JSON)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Defines a set of strings for localizing plugin names and user-facing text. It supports multiple languages, with a default string for fallback. This structure is used with `LocalizedString`.

```json
{
	"name": "my-plugin",
	"strings": {
		"my-plugin": {
			"default": "My Plugin",
			"it": "Il mio Plugin",
			"fr": "Mon Plugin"
		}
	}
}
```

---

### getFolder

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider

Opens a folder picker dialog to select a folder, providing access to its contents for read-write operations.

````APIDOC
## GET /getFolder

### Description
Gets a folder from the file system via a folder picker dialog. The files and folders within can be accessed via `Folder#getEntries`. Any files within are read-write. If the user dismisses the picker, `null` is returned instead.

### Method
GET

### Endpoint
`/getFolder`

### Parameters
#### Query Parameters
- **options** (`any`) - Optional - Options for folder selection.
  - **initialDomain** (`Symbol`) - Optional - The preferred initial location of the file picker. If not defined, the most recently used domain from a file picker is used instead.

### Returns
`Promise<Folder | null>` - The selected folder or `null` if no folder is selected.

### Request Example
```javascript
const folder = await fs.getFolder();
const myNovel = (await folder.getEntries()).filter(entry => entry.name.indexOf('novel') > 0);
const text = await myNovel.read();
````

````

--------------------------------

### XMPIterator.next()

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPIterator

Retrieves the next item in the metadata during iteration. Returns an XMPProperty object or null if there are no more items.

```APIDOC
## next()

### Description
Retrieves the next item in the metadata.

### Method
`next()`

### Endpoint
N/A (Method of XMPIterator object)

### Parameters
N/A

### Request Example
```javascript
let currentProperty = iterator.next();
while (currentProperty !== null) {
    // Process currentProperty
    currentProperty = iterator.next();
}
````

### Response

#### Success Response (200)

- **XMPProperty** (object) | **null** - The next XMPProperty object in the iteration, or null if the iteration is complete.

````

--------------------------------

### UXP Command and Panel Retrieval Functions in JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Entry%20Points/EntryPoints

These JavaScript functions allow retrieval of panel and command objects by their IDs. `getPanel(id)` returns a UxpPanelInfo object or null if the ID is invalid. `getCommand(id)` returns a UxpCommandInfo object or null if the ID is invalid. These are essential for interacting with plugin elements.

```javascript
/**
 * Get panel with specified id
 * @param {String} id - panel id
 * @returns {UxpPanelInfo|null} - panel object for a valid id null for an invalid id
 */
function getPanel(id) {
    // ... implementation
}

/**
 * Get command with specified id
 * @param {String} id - command id
 * @returns {UxpCommandInfo|null} - command object for a valid id null for an invalid id
 */
function getCommand(id) {
    // ... implementation
}
````

---

### Pointer Capture

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHtmlElement

Methods for managing pointer capture for events.

````APIDOC
## setPointerCapture(pointerId)

### Description
Sets pointer capture for the element. This means that all future pointer events will be dispatched to this element, even if the pointer (e.g., mouse cursor) moves outside its boundaries. This implementation does not dispatch the `gotpointercapture` event on the element.

### Method
POST (conceptually, as it changes event handling)

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage:
// Assume 'e' is a pointer event object from a 'pointerdown' event
// element.setPointerCapture(e.pointerId);
````

### Response

#### Success Response (200)

None (modifies pointer capture state)

#### Response Example

None

### Error Handling

- **DOMException**: Thrown if the element is not connected to the DOM.

### Since

v7.1

````

```APIDOC
## releasePointerCapture(pointerId)

### Description
Releases pointer capture for the element. This stops all future pointer events from being dispatched exclusively to this element. This implementation does not dispatch the `lostpointercapture` event on the element.

### Method
POST (conceptually, as it changes event handling)

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage:
// Assume 'e' is a pointer event object from a 'pointerup' event
// element.releasePointerCapture(e.pointerId);
````

### Response

#### Success Response (200)

None (modifies pointer capture state)

#### Response Example

None

### Since

v7.1

````

--------------------------------

### JSON for Premiere Pro UXP Plugin Manifest

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-modal-dialogs

The plugin manifest file (manifest.json) for an Adobe Premiere Pro UXP plugin. It defines the plugin's name, ID, version, host application, and entry points for panels and commands.

```json
{
  "id": "com.example.premierepro.dialogdemo",
  "version": "1.0.0",
  "name": "Dialog Demo",
  "host": {
    "appId": "PHLC",
    "minVersion": "15.0.0"
  },
  "panels": [
    {
      "id": "com.example.premierepro.dialogdemo.panel",
      "name": "Dialog Demo Panel",
      "process": "main.js",
      "container": "#panel-container",
      "menuItems": [
        {
          "id": "showDialog",
          "label": "Show Dialog",
          "commandId": "showDialogCommand"
        }
      ]
    }
  ],
  "entrypoints": [
    {
      "id": "showDialogCommand",
      "container": "#dialog-container",
      "title": "Set Sequence Dimensions",
      "filePath": "./dialog.html"
    }
  ]
}

````

---

### IntersectionObserverEntry Constructor

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20DOM/IntersectionObserverEntry

Initializes a new IntersectionObserverEntry object with provided intersection data.

````APIDOC
## IntersectionObserverEntry(intersectionObserverEntryInit)

### Description
Creates an instance of IntersectionObserverEntry.

### Method
Constructor

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **intersectionObserverEntryInit** (`IntersectionObserverEntryInit`) - Required - An object containing the initial values for the entry.

### Request Example
```json
{
  "intersectionObserverEntryInit": {
    "time": 1678886400000,
    "rootBounds": {"x": 0, "y": 0, "width": 800, "height": 600, "top": 0, "right": 800, "bottom": 600, "left": 0},
    "boundingClientRect": {"x": 100, "y": 100, "width": 200, "height": 150, "top": 100, "right": 300, "bottom": 250, "left": 100},
    "intersectionRect": {"x": 100, "y": 100, "width": 100, "height": 75, "top": 100, "right": 200, "bottom": 175, "left": 100},
    "isIntersecting": true,
    "intersectionRatio": 0.5,
    "target": "<element>"
  }
}
````

### Response

#### Success Response (200)

- **IntersectionObserverEntry** (`IntersectionObserverEntry`) - The newly created IntersectionObserverEntry object.

#### Response Example

```json
{
	"time": 1678886400000,
	"rootBounds": { "x": 0, "y": 0, "width": 800, "height": 600, "top": 0, "right": 800, "bottom": 600, "left": 0 },
	"boundingClientRect": { "x": 100, "y": 100, "width": 200, "height": 150, "top": 100, "right": 300, "bottom": 250, "left": 100 },
	"intersectionRect": { "x": 100, "y": 100, "width": 100, "height": 75, "top": 100, "right": 200, "bottom": 175, "left": 100 },
	"isIntersecting": true,
	"intersectionRatio": 0.5,
	"target": "<element>"
}
```

````

--------------------------------

### ProjectSettings - getIngestSettings

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/projectsettings

Retrieves the current ingest settings for a project.

```APIDOC
## GET /websites/developer_adobe_premiere-pro_uxp/ProjectSettings/getIngestSettings

### Description
Returns project ingest settings.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/ProjectSettings/getIngestSettings

### Parameters
#### Query Parameters
- **project** (_Project_) - Required - The project for which to retrieve ingest settings.

### Request Example
GET /websites/developer_adobe_premiere-pro_uxp/ProjectSettings/getIngestSettings?project=project_object

### Response
#### Success Response (200)
- **ingestSettings** (_IngestSettings_) - The current ingest settings for the project.

#### Response Example
{
  "ingestSettings": "ingest_settings_object"
}
````

---

### Element Interaction API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLDialogElement

APIs for triggering clicks on elements.

```APIDOC
## click()

### Description
Programmatically triggers a click event on the element.

### Method
`click`

### Parameters
None
```

---

### Create and Display ImageBlob from Uncompressed Data

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/ImageBlob

This snippet demonstrates how to create an ImageBlob from raw pixel data (uncompressed image) and display it in an HTML `<img>` element. It involves preparing an ArrayBuffer with pixel values, defining metadata in an options object, instantiating ImageBlob, generating a URL, setting it as the `src` for an image element, and revoking the URL when no longer needed. Dependencies include a standard HTML structure and the `ImageBlob` constructor.

```html
<!DOCTYPE html>
<html>
	<head>
		<script src="index.js"></script>
	</head>
	<style>
		body {
			background-color: whitesmoke;
			padding: 0 16px;
		}

		#image,
		container {
			margin: 8px;
			display: flex;
			flex-direction: row;
			justify-content: flex-start;
		}
	</style>

	<body>
		<div class="container">
			<sp-button
				id="pixel-btn"
				variant="secondary"
				quiet
				>Paint image</sp-button
			>
			<img id="image" />
		</div>
	</body>
</html>
```

```javascript
// Updating HTML with ImageBlob

//Creating ImageBlob by creating the options Object seperatly and then pass the Object as argument
function getPixel() {
	const imageMetaData = {
		width: 8,
		height: 8,
		colorSpace: "RGB",
		colorProfile: "",
		pixelFormat: "RGB",
		components: 3,
		componentSize: 8,
		hasAlpha: false, // Alpha is set to false
		type: "image/uncompressed",
	};

	let buffer = new ArrayBuffer(imageMetaData.width * imageMetaData.height * 3);
	let colorArrayView = new Uint8Array(buffer);
	for (let i = 0; i < colorArrayView.length / 4; i++) {
		// here we are creating a red image, update values to see the variations
		colorArrayView[i * 3] = 255; // Red Component
		colorArrayView[i * 3 + 1] = 0; // Green Component
		colorArrayView[i * 3 + 2] = 0; // Blue Component
	}
	let imageBlob = new ImageBlob(colorArrayView, imageMetaData);
	// Generate url which can be used as src on HTMLImageElement
	const url = URL.createObjectURL(imageBlob);
	// ensure that there is a HTMLImageElement in the Document with id `image`.
	const imageElement = document.getElementById("image");
	imageElement.src = url;

	// revoke(destroy image from the memory) when url is no more required.
	URL.revokeObjectURL(url);
}
document.addEventListener("DOMContentLoaded", () => {
	document.getElementById("pixel-btn").addEventListener("click", getPixel);
});
```

---

### CloseEvent Constructor

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Events/CloseEvent

Creates a new instance of the CloseEvent. It accepts parameters to define the code, reason, and cleanliness of the closure.

````APIDOC
## CloseEvent(code, reason, wasClean)

### Description
Creates an instance of CloseEvent.

### Parameters
#### Path Parameters
- **code** (*)
- **reason** (*)
- **wasClean** (*)

### Request Example
```javascript
const closeEvent = new CloseEvent('close', {
  code: 1000,
  reason: 'Normal closure',
  wasClean: true
});
````

### Response

#### Success Response (200)

- **CloseEvent** (object) - An instance of the CloseEvent.

#### Response Example

```json
{
	"message": "CloseEvent instance created successfully"
}
```

````

--------------------------------

### Open File Asynchronously - JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/fs

Opens or creates a file asynchronously. It returns a file descriptor (fd) which can be used for subsequent file operations. Supports optional flags and modes similar to Node.js file system operations. If no callback is provided, it returns a Promise.

```javascript
const fd = await fs.open("plugin-data:/fileToRead.txt", "r");
````

---

### HTMLWebViewElement Usage

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

This section describes how to add and configure the HTMLWebViewElement in your UXP plugin. It includes attributes for setting properties like ID, height, and source URL.

````APIDOC
## HTMLWebViewElement

### Description

The `HTMLWebViewElement` is a component that allows you to embed web content directly within your UXP plugins. It functions as a self-contained browser window, enabling the display and interaction with web pages using JavaScript.

### Method

N/A (This is a component, not an endpoint)

### Endpoint

N/A (This is a component, not an endpoint)

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```html
<webview id="webviewsample" width="100%" height="360px" src="https://www.adobe.com" uxpAllowInspector="true"></webview>
````

### Response

N/A (This is a client-side component)

#### Success Response (200)

N/A

#### Response Example

N/A

````

--------------------------------

### Enable SWC Support in manifest.json

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/swc

Configure your plugin's manifest.json file to enable Spectrum Web Components (SWC) support. This requires manifestVersion 5 or above and setting the 'enableSWCSupport' feature flag to true.

```json
{
  "manifestVersion": 5,
  "host": [
    {
      "app": "PS",
      "minVersion": "24.4"
    }
  ],
  "featureFlags": {
    "enableSWCSupport": true
  }
}
````

---

### Querying Elements using Selectors in JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLElement

Methods to find elements within the DOM using CSS selectors. `querySelector` returns the first matching element, while `querySelectorAll` returns a NodeList of all matches.

```javascript
// Select the first element matching the CSS selector
const firstDiv = document.querySelector("div.some-class");

// Select all elements matching the CSS selector
const allParagraphs = document.querySelectorAll("p.content");
```

---

### Pointer Event Handling

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLMenuElement

APIs for managing pointer capture and checking pointer capture status on an element.

````APIDOC
## releasePointerCapture(pointerId)

### Description
Releases pointer capture for the element. This implementation does not dispatch the `lostpointercapture` event on the element.

### Method
(Implicitly called on an element instance)

### Endpoint
N/A (Method on element instance)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```javascript
// Assuming 'element' is a reference to an HTML element
element.releasePointerCapture(pointerId);
````

### Response

#### Success Response (200)

N/A (Method does not return a value)

#### Response Example

N/A

---

## hasPointerCapture(pointerId)

### Description

Checks if the element has pointer capture for the specified pointer.

### Method

(Implicitly called on an element instance)

### Endpoint

N/A (Method on element instance)

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```javascript
// Assuming 'element' is a reference to an HTML element
const hasCapture = element.hasPointerCapture(pointerId);
```

### Response

#### Success Response (200)

- **boolean** - True if the element has pointer capture for the specified pointer, false otherwise.

#### Response Example

```json
true
```

````

--------------------------------

### VideoFilterFactory - createComponent

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/videofilterfactory

Creates a new video filter component based on the provided match name. This allows for dynamic instantiation of video filters.

```APIDOC
## POST /videoFilterFactory/createComponent

### Description
Creates a new video filter component based on the input matchName.

### Method
POST

### Endpoint
/videoFilterFactory/createComponent

### Parameters
#### Query Parameters
- **matchName** (string) - Required - The match name of the component to create, example 'PR.ADBE Solarize', 'AE.ADBE Mosaic' etc..

### Request Example
```json
{
  "matchName": "PR.ADBE Solarize"
}
````

### Response

#### Success Response (200)

- **VideoFilterComponent** - The newly created video filter component.

#### Response Example

```json
{
	"component": "<instance of VideoFilterComponent>"
}
```

````

--------------------------------

### DOM Traversal and Matching

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLMenuElement

Methods for querying and traversing the DOM tree, including finding closest elements and checking element matches.

```APIDOC
## closest(selectorString)

### Description
Returns the closest ancestor of the current element (which is itself) that matches the specified group of selectors.

### Method
(Implicitly called on an element instance)

### Endpoint
N/A (Method on element instance)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```javascript
// Assuming 'element' is a reference to an HTML element
const closestElement = element.closest('.some-class');
````

### Response

#### Success Response (200)

- **Element** - The first ancestor element that matches the selectors, or null if none match.

#### Response Example

```html
<div class="some-class">...</div>
```

---

## matches(selectorString)

### Description

Tests whether an element is an instance of a class or matches a CSS selector.

### Method

(Implicitly called on an element instance)

### Endpoint

N/A (Method on element instance)

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```javascript
// Assuming 'element' is a reference to an HTML element
const isMatch = element.matches("div.container");
```

### Response

#### Success Response (200)

- **boolean** - True if the element matches the selector, false otherwise.

#### Response Example

```json
true
```

````

--------------------------------

### Read Directory Asynchronously (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/fs

Reads a directory asynchronously to list its contents (files and subdirectories). Returns a Promise that resolves with an array of strings representing the paths of the contents. Requires a string path and accepts an optional callback function.

```javascript
const paths = await fs.readdir("plugin-data:/dirToRead");
````

---

### Event Handling

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLTextAreaElement

Methods for managing event listeners on elements.

````APIDOC
## addEventListener(eventName, callback, options)

### Description
Attaches an event listener to the element.

### Method
(Implicitly called on an element instance)

### Endpoint
N/A (Instance Method)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage:
// element.addEventListener('click', handleClick, { capture: true });
````

### Response

#### Success Response (200)

None (void)

#### Response Example

None

````

```APIDOC
## removeEventListener(eventName, callback, options)

### Description
Removes an event listener from the element.

### Method
(Implicitly called on an element instance)

### Endpoint
N/A (Instance Method)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage:
// element.removeEventListener('click', handleClick, { capture: true });
````

### Response

#### Success Response (200)

None (void)

#### Response Example

None

````

```APIDOC
## dispatchEvent(event)

### Description
Dispatches a synthetic event to the element.

### Method
(Implicitly called on an element instance)

### Endpoint
N/A (Instance Method)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage:
// const clickEvent = new Event('click');
// element.dispatchEvent(clickEvent);
````

### Response

#### Success Response (200)

- **boolean** - `true` if the event is dispatched successfully, `false` otherwise.

#### Response Example

```json
true
```

````

--------------------------------

### Enable Alerts in Manifest - JSON

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20DOM/alert

Configuration snippet for the UXP plugin manifest file (v5) to enable the 'alert()' function. This is required for plugins since UXP v7.4.

```json
{
  "featureFlags": {
    "enableAlerts": true
  }
}
````

---

### Attach Shadow DOM (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLScriptElement

Attaches a shadow DOM tree to the specified element, returning a reference to its ShadowRoot. This feature requires enabling `enableSWCSupport` via a feature flag in the plugin manifest. The `init` parameter specifies mode, focus delegation, and slot assignment.

```javascript
function attachShadow(init) {
	// Implementation details...
}
```

---

### getFileForOpening

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider

Retrieves files from the file system for opening. Files are read-only and can be opened individually or in multiples.

````APIDOC
## GET /getFileForOpening

### Description
Gets a file (or files) from the file system provider for the purpose of opening them. Files are read-only. Multiple files can be returned if the `allowMultiple` option is true.

### Method
GET

### Endpoint
`/getFileForOpening`

### Parameters
#### Query Parameters
- **options** (`*`) - Required - Options object for file opening.
  - **initialDomain** (`Symbol`) - Optional - The preferred initial location of the file picker. If not defined, the most recently used domain from a file picker is used instead.
  - **types** (`Array<string>`) - Optional - Defaults to `['.*']`. An array of file types that the file open picker displays.
  - **initialLocation** (`File` | `Folder`) - Optional - The initial location of the file picker. You can pass an existing file or folder entry to suggest the picker to start at this location. If this is a file entry, then the method will pick its parent folder as the initial location. This will override the `initialDomain` option.
  - **allowMultiple** (`boolean`) - Optional - Defaults to `false`. If true, multiple files can be selected.

### Returns
`Promise<File|Array<File>>` - Based on `allowMultiple`. Returns an empty array or null if no file was selected.

### Request Example (Single File)
```javascript
const folder = await fs.getFolder({initialDomain: domains.userDocuments});
const file = await fs.getFileForOpening({initialLocation: folder});
if (!file) {
    // no file selected
    return;
}
const text = await file.read();
````
