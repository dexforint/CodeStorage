### Request Example (Multiple Files)

```javascript
const files = await fs.getFileForOpening({ allowMultiple: true, types: fileTypes.images });
if (files.length === 0) {
	// no files selected
}
```

````

--------------------------------

### Open File/Folder Path with Shell API (UXP)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/shell/Shell

Opens a specified file or folder path using the system's default application. This function is restricted to files within the UXP App sandbox. It returns a promise that resolves with an empty string on success or an error message on failure. Requires UXP Manifest v5.

```javascript
const { shell } = require('uxp');

// Example for MacOS
shell.openPath("/Users/[username]/Downloads");
shell.openPath("/Users/[username]/sample.txt");

// Example for Windows
shell.openPath("C:\Users\[username]\Downloads");
shell.openPath("C:\Users\[username]\AppData\Local\...\sample.txt");
````

---

### Connect to WebSocket Server

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/network

Establishes a WebSocket connection to a specified server URL. It manages connection states, sends a message upon successful connection, and logs incoming messages and errors. Supports live updates for real-time applications. Requires network permissions for the target domain.

```javascript
let socket;

async function connectToServer() {
	try {
		if (socket) {
			console.log("🔌 Disconnecting existing socket...");
			socket.close();
			socket = null;
			return;
		}

		socket = new WebSocket("wss://javascript.info/article/websocket/demo/hello");

		socket.onopen = () => {
			console.log("✅ WebSocket connection established");
			socket.send("Hello from Premiere plugin!");
		};

		socket.onmessage = (event) => {
			console.log(`📩 Message from server: ${event.data}`);
		};

		socket.onerror = (err) => {
			console.error("⚠️ WebSocket error:", err);
		};

		socket.onclose = () => {
			console.log("Connection closed");
			socket = null;
		};
	} catch (e) {
		console.error("Failed to connect via WebSocket:", e);
	}
}
```

---

### Render Basic Action Button - HTML

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-action-button

Renders a basic action button with text content. This is the simplest form of the sp-action-button.

```html
<sp-action-button>An Action</sp-action-button>
```

---

### Spectrum UXP sp-heading for Theme-Aware Headings

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-html/Hierarchy/h5

This snippet shows how to achieve a theme-aware heading using the Spectrum UXP 'sp-heading' component. This component is recommended when you need headings that automatically adapt to the application's color scheme, providing a consistent user experience.

```html
<sp-heading size="XXS">Hello, World</sp-heading>
```

---

### Clipboard Operations API

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/clipboard

UXP provides clipboard APIs that let your plugin read from and write to the system clipboard, enabling users to copy, paste, and share content between your plugin and other applications.

````APIDOC
## Clipboard Operations API

This API allows plugins to interact with the system clipboard for reading and writing text data.

### Permissions

To access the clipboard, you must declare permissions in your `manifest.json` file. The available permissions are:

- `"read"`: Allows the plugin to only read data from the clipboard.
- `"readAndWrite"`: Allows the plugin to both read from and write to the clipboard.

Choose the least permissive option that meets your plugin's needs.

### Core Methods

The clipboard is accessed through `navigator.clipboard`:

- `setContent(data)`: Writes data to the clipboard.
- `getContent()`: Reads data from the clipboard.

Both methods work with MIME type objects, where keys represent data formats (e.g., `"text/plain"`) and values contain the actual content.

## Example: Writing to the Clipboard

This example demonstrates how to copy text to the system clipboard.

### `index.js`
```javascript
// Copy formatted text to the clipboard ✂️
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.setContent({
      "text/plain": text
    });
    console.log("✅ Text copied to clipboard");
  } catch (err) {
    console.error("❌ Failed to copy to clipboard:", err);
  }
}

// Example usage
copyToClipboard("Welcome to UXP for Premiere!");
````

### `manifest.json`

```json
{
	// ...
	"requiredPermissions": {
		"clipboard": "readAndWrite"
	}
	// ...
}
```

## Example: Reading from the Clipboard

This example shows how to read content from the system clipboard.

### `index.js`

```javascript
// Paste text from the clipboard 📋
async function pasteFromClipboard() {
	try {
		const clipboardData = await navigator.clipboard.getContent();

		if (clipboardData["text/plain"]) {
			console.log(`Pasted text: ${clipboardData["text/plain"]}`);
			return clipboardData["text/plain"];
		} else {
			console.log("⚠️ No text data found on clipboard");
			return null;
		}
	} catch (err) {
		console.error("❌ Failed to read from clipboard:", err);
	}
}

// Example usage
pasteFromClipboard();
```

### `manifest.json`

```json
{
	// ...
	"requiredPermissions": {
		"clipboard": "read"
	}
	// ...
}
```

## Example: Copy and Paste Together

This example demonstrates copying and then transforming clipboard text.

### `index.js`

```javascript
// Transform clipboard text to uppercase
async function transformClipboardText() {
	try {
		// Read current clipboard content
		const data = await navigator.clipboard.getContent();

		if (data["text/plain"]) {
			const originalText = data["text/plain"];
			const transformedText = originalText.toUpperCase();

			// Write the transformed text back
			await navigator.clipboard.setContent({
				"text/plain": transformedText,
			});

			console.log(`✅ Transformed: "${originalText}" → "${transformedText}"`);
		} else {
			console.log("⚠️ No text found on clipboard");
		}
	} catch (err) {
		console.error("Failed to transform clipboard text:", err);
	}
}

// Example usage
transformClipboardText();
```

### `manifest.json` (for copy and paste together)

```json
{
	// ...
	"requiredPermissions": {
		"clipboard": "readAndWrite"
	}
	// ...
}
```

````

--------------------------------

### AddTransitionOptions Properties

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/addtransitionoptions

This section describes the properties of the AddTransitionOptions class, which allow you to configure various aspects of a transition.

```APIDOC
## AddTransitionOptions Properties

### Description
Properties to configure transition settings such as application side, force single-sided application, alignment, and duration.

### Properties
- **applyToStart** (boolean) - R - Determines if the transition should be applied to the start or end of the track item. Available from version 25.0.
- **forceSingleSided** (boolean) - R - Determines if the transition should be applied to one or both sides. Available from version 25.0.
- **transitionAlignment** (number) - R - Specifies the alignment of the transition. Available from version 25.0.
- **duration** (TickTime) - R - Sets the duration of the transition. Available from version 25.0.
````

---

### Send POST Request (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

Shows how to send a POST request with form-urlencoded data to a server and log the response. It configures the request's onreadystatechange event to handle the response when the request is complete and successful.

```javascript
const xhr = new XMLHttpRequest();
xhr.onreadystatechange = () => {
	if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
		console.log(xhr.responseText);
	}
};
xhr.open("POST", "https://www.myserver.com");
xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
xhr.send("foo=bar&lorem=ipsum");
```

---

### Write File Asynchronously - UXP FSAPI

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/fs

Writes data to a specified file path asynchronously. Data can be a string, ArrayBuffer, or ArrayBufferView. Supports options for file flags and encoding. Returns a Promise with the number of bytes written. If no callback is provided, a Promise is returned.

```javascript
const bufLen = await fs.writeFile("plugin-data:/binaryFile.obj", new Uint8Array([1, 2, 3]));
```

```javascript
const strLen = await fs.writeFile("plugin-data:/textFile.txt", "It was a dark and stormy night.\n", { encoding: "utf-8" });
```

---

### Build Plugin with npm

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/typescript-support

Executes the build script defined in package.json to compile TypeScript code into JavaScript.

```bash
npm run build
```

---

### Element Querying

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLSelectElement

Methods for finding elements within the DOM.

```APIDOC
## getElementsByClassName(name)

### Description
Returns a NodeList of all child elements of the current element that have the specified class name.

### Method
`getElementsByClassName`

### Parameters
#### Path Parameters
- **name** (string) - The class name to search for.

### Returns
`NodeList`
```

```APIDOC
## getElementsByTagName(name)

### Description
Returns a NodeList of all child elements of the current element that have the specified tag name.

### Method
`getElementsByTagName`

### Parameters
#### Path Parameters
- **name** (string) - The tag name to search for.

### Returns
`NodeList`
```

```APIDOC
## querySelector(selector)

### Description
Returns the first element within the document (or within the element's subtree) that matches the specified group of selectors.

### Method
`querySelector`

### Parameters
#### Path Parameters
- **selector** (string) - A string containing one or more CSS selectors to match.

### Returns
`Element`
```

```APIDOC
## querySelectorAll(selector)

### Description
Returns a NodeList representing a list of the document's elements that match the specified group of selectors.

### Method
`querySelectorAll`

### Parameters
#### Path Parameters
- **selector** (string) - A string containing one or more CSS selectors to match.

### Returns
`NodeList`
```

---

### Read Directory Synchronously (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/fs

Reads a directory synchronously to list its contents (files and subdirectories). Returns an array of strings representing the paths of the contents. Requires a string path as input.

```javascript
const paths = fs.readdirSync("plugin-data:/dirToRead");
```

---

### Create Shadow DOM Tree (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20DOM/Element

Explains how to attach a shadow DOM tree to an element using the `attachShadow` method. Note that this feature requires enabling the `enableSWCSupport` flag in the plugin manifest. It returns a reference to the ShadowRoot.

```javascript
// Ensure enableSWCSupport is true in plugin manifest
const shadowRoot = element.attachShadow({ mode: "open" });
```

---

### Pointer Capture API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLTextAreaElement

Manage pointer capture for elements to control mouse and touch interactions.

````APIDOC
## releasePointerCapture(pointerId)

### Description
Releases pointer capture for the element. This implementation does not dispatch the `lostpointercapture` event on the element.

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
// element.releasePointerCapture(pointerId);
````

### Response

#### Success Response (200)

None (void)

#### Response Example

None

````

```APIDOC
## hasPointerCapture(pointerId)

### Description
Checks if the element has pointer capture for the specified pointer.

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
// const hasCapture = element.hasPointerCapture(pointerId);
````

### Response

#### Success Response (200)

- **boolean** - True if the element has pointer capture for the specified pointer, false otherwise.

#### Response Example

```json
true
```

````

--------------------------------

### Handling Input Events - UXP JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-textfield

Demonstrates how to listen for and respond to input changes in a text field using JavaScript. The `input` event captures real-time value updates, logging them to the console.

```javascript
document.querySelector(".yourTextField").addEventListener("input", evt => {
    console.log(`New value: ${evt.target.value}`);
})
````

---

### POST Request with JSON Payload using fetch() in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/network

Illustrates how to perform a POST request with a JSON payload using the `fetch` API in UXP. This is common for sending data to web APIs. Ensure the target domain is allow-listed in `manifest.json` and correctly handle response parsing and errors.

```javascript
async function postUserData(user) {
	try {
		const response = await fetch("https://api.example.com/users", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(user),
		});

		if (!response.ok) throw new Error(`Server error: ${response.status}`);
		const result = await response.json();
		console.log("✅ User created:", result);
	} catch (err) {
		console.error("Failed to post user data:", err);
	}
}

// Example usage
postUserData({ name: "Jamie", role: "Editor" });
```

---

### EventTarget Methods

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLinkElement

This section covers methods for handling events on EventTarget objects, including adding and removing event listeners, and dispatching events.

````APIDOC
## addEventListener(eventName, callback, options)

### Description
Attaches an event listener to the EventTarget.

### Method
POST

### Endpoint
EventTarget.addEventListener

### Parameters
#### Request Body
- **eventName** (`string`) - Required - The name of the event to listen for.
- **callback** (`function`) - Required - The function to call when the event is fired.
- **options** (`boolean` or `Object`) - Optional - A boolean value denoting capture value or an options object. Currently supports only capture in options object ({ capture: bool_value }).

### Request Example
```json
{
  "eventName": "click",
  "callback": "function(event) { console.log('Clicked!'); }",
  "options": {"capture": true}
}
````

### Response

#### Success Response (200)

No content returned on success.

````

```APIDOC
## removeEventListener(eventName, callback, options)

### Description
Removes an event listener from the EventTarget.

### Method
DELETE

### Endpoint
EventTarget.removeEventListener

### Parameters
#### Request Body
- **eventName** (`string`) - Required - The name of the event to remove the listener for.
- **callback** (`function`) - Required - The event listener function to remove.
- **options** (`boolean` or `Object`) - Optional - A boolean value denoting capture value or an options object. Must match the options used when adding the listener.

### Request Example
```json
{
  "eventName": "click",
  "callback": "function(event) { console.log('Clicked!'); }",
  "options": {"capture": true}
}
````

### Response

#### Success Response (200)

No content returned on success.

````

```APIDOC
## dispatchEvent(event)

### Description
Dispatches an `Event` to this `EventTarget`, setting the `target` and `currentTarget` properties to this `EventTarget`. It also sets the `eventPhase` to `AT_TARGET` and calls the event listeners registered for this event.

### Method
POST

### Endpoint
EventTarget.dispatchEvent

### Parameters
#### Request Body
- **event** (`Event`) - Required - The event to dispatch.

### Request Example
```json
{
  "event": {
    "type": "customEvent",
    "detail": {"message": "Hello!"}
  }
}
````

### Response

#### Success Response (200)

- **boolean** - True if the event was dispatched successfully, false otherwise.

### Response Example

```json
true
```

````

--------------------------------

### HTMLVideoElement: Setting Volume

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLVideoElement

Illustrates how to set the volume of a video element and listen for the 'volumechange' event. This is essential for providing user control over audio levels in media playback.

```javascript
vid.volume = 0.5; // Set volume to 50%
vid.addEventListener("volumechange", (event) => {
    console.log("Volume changed to: " + vid.volume);
});
````

---

### Dialog Management API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLDialogElement

APIs for showing and closing dialogs, including non-modal and modal types.

````APIDOC
## show([options])

### Description
Shows a non-modal dialog.

### Method
`show`

### Parameters
#### Query Parameters
- **options** (object) - Optional - Options for the show method.
  - **options.anchorOffset** (object) - Offset from the anchor for the initial positioning of the dialog.
    - **options.anchorOffset.top** (number) - Top offset from the anchor for the initial positioning of the dialog.
    - **options.anchorOffset.left** (number) - Left offset from the anchor for the initial positioning of the dialog.

### Response Example
(No specific response example provided for show method itself, but it modifies the UI.)

---

## showModal()

### Description
Shows a modal dialog. Returns a Promise that resolves when the dialog is closed.

### Method
`showModal`

### Parameters
None

### Response
#### Success Response (Promise Resolved)
- **returnValue** (*) - The value passed to the `close()` method or the value from the submit button.

#### Error Response (Promise Rejected)
- **error** (object) - Details about why the dialog was closed or rejected.
  - **error.code** (number) - One of the values from `HTMLDialogElement.rejectionReasons`.

### Response Example
```javascript
showModal().then(returnValue => {
  console.log('Dialog closed with:', returnValue);
}).catch(error => {
  console.error('Dialog rejected:', error);
});
````

---

## close([returnValue])

### Description

Closes the dialog and optionally sets a return value.

### Method

`close`

### Parameters

#### Query Parameters

- **returnValue** (\*) - Optional - The value to be returned when the dialog is closed.

### Request Example

```javascript
close("someValue");
```

````

--------------------------------

### Transcript Import/Export API

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/transcript

Provides methods for creating import transcript actions, exporting transcripts to JSON, and importing transcripts from JSON.

```APIDOC
## POST /websites/developer_adobe_premiere-pro_uxp/transcript/createImportTextSegmentsAction

### Description
Creates an action to import external transcripts into a ClipProjectItem.

### Method
POST

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/transcript/createImportTextSegmentsAction

### Parameters
#### Request Body
- **textSegments** (_TextSegments_) - Required - The text segments to import.
- **clipProjectItem** (_ClipProjectItem_) - Required - The clip project item to import into.

### Response
#### Success Response (200)
- **action** (_Action_) - The created action object.

## POST /websites/developer_adobe_premiere-pro_uxp/transcript/exportToJSON

### Description
Exports transcripts within a ClipProjectItem as a JSON string.

### Method
POST

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/transcript/exportToJSON

### Parameters
#### Request Body
- **clipProjectItem** (_ClipProjectItem_) - Required - The clip project item containing the transcripts.

### Response
#### Success Response (200)
- **jsonString** (_string_) - The exported JSON string of the transcripts.

## POST /websites/developer_adobe_premiere-pro_uxp/transcript/importFromJSON

### Description
Initializes a TextSegments object from a JSON string.

### Method
POST

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/transcript/importFromJSON

### Parameters
#### Request Body
- **jsonString** (_string_) - Required - The JSON string to import from.

### Response
#### Success Response (200)
- **textSegments** (_TextSegments_) - The initialized TextSegments object.
````

---

### Displaying sp-icon with name attribute

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-icon

Demonstrates how to render a Spectrum icon using its 'name' attribute. The 'name' attribute specifies which icon to display, such as 'ui:Magnifier'. This is a fundamental usage of the sp-icon component.

```html
<sp-icon name="ui:Magnifier"></sp-icon>
```

---

### Set and Release Pointer Capture - JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLVideoElement

Demonstrates how to capture and release pointer events on an element using setPointerCapture and releasePointerCapture. This is useful for implementing drag-and-drop functionality or other pointer-driven interactions. It requires adding and removing event listeners for pointer movement.

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
```

---

### Implement Webview Communication

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Demonstrates how to send and receive messages between a plugin and a webview using the message API. This allows for bidirectional communication.

```javascript
// In the plugin:
const webview = document.querySelector("webview");
webview.addEventListener("message", (event) => {
	console.log("Received message from webview:", event.data);
	webview.postMessage("Hello from the plugin!");
});

// In the webview:
window.addEventListener("message", (event) => {
	console.log("Received message from plugin:", event.data);
	window.uxpHost.postMessage("Hello from the webview!");
});
```

---

### Enable User Info Permission in manifest.json

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/User%20Information

This JSON snippet shows how to add the `enableUserInfo` permission to your plugin's `manifest.json` file. This permission is required to access user information APIs.

```json
{
	"requiredPermissions": {
		"enableUserInfo": true
	}
}
```

---

### Basic Plugin HTML Structure

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/css-styling

This HTML file provides the basic structure for the UXP plugin, including a heading and a content area. It uses Spectrum Web Components ('sp-heading', 'sp-body') and a div with the class 'plugin-body' which is styled by the CSS.

```html
<body>
	<sp-heading>My Plugin</sp-heading>
	<div class="plugin-body">
		<sp-body>This is the main content area.</sp-body>
	</div>
</body>
```

---

### Scrolling Methods

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHeadElement

Provides methods for scrolling elements to specific positions or into view.

````APIDOC
## scrollTo(xOrOptions, y)

### Description
Scrolls the element to the new x and y positions. If options object is used with behavior: "smooth" then the element is smoothly scrolled.

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

### Request Example
```javascript
// Example using options object for smooth scrolling
// element.scrollTo({ top: 100, left: 0, behavior: 'smooth' });

// Example using x and y coordinates
// element.scrollTo(0, 100);
````

### Response

#### Success Response (200)

N/A

#### Response Example

N/A

## scrollIntoView(alignToTop)

### Description

Scrolls the element's ancestors so that the element is visible to the user.

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

### Request Example

```javascript
// Scrolls the element into view, aligning the top edge
// element.scrollIntoView(true);

// Scrolls the element into view, aligning the top edge if possible
// element.scrollIntoView();
```

### Response

#### Success Response (200)

N/A

#### Response Example

N/A

## scrollIntoViewIfNeeded()

### Description

Scrolls the element's ancestors so that the element is visible to the user, only if it is not already visible.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

N/A

### Request Example

```javascript
// element.scrollIntoViewIfNeeded();
```

### Response

N/A

````

--------------------------------

### Strings Definition

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Represents a set of strings for localizing plugin names and other user-facing text.

```APIDOC
## StringsDefinition

### Description
Represents a set of strings used to localize the plugin name and other user-facing strings. `StringsDefinition` keys can be used anywhere where `LocalizedString` is supported.

### Example
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
````

````

--------------------------------

### Enable Experimental Feature Flags

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Allows plugins to utilize experimental features. This includes support for CSS variables in SVG fills, Spectrum Web Components, and native alert/prompt/confirm dialogs.

```json
{
  "enableFillAsCustomAttribute": true,
  "enableSWCSupport": true,
  "enableAlerts": true
}
````

---

### Element Dragging with Pointer Events (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLScriptElement

Demonstrates how to implement basic element dragging functionality using pointer events in JavaScript. It captures pointer events on `pointerdown`, updates element position on `pointermove`, and releases capture on `pointerup`. Includes basic HTML and CSS for a draggable div.

```html
<style>
	div {
		width: 140px;
		height: 50px;
		display: flex;
		align-items: center;
		justify-content: center;
		background: #fbe;
		position: absolute;
	}
</style>
<div id="slider">SLIDE ME</div>
```

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
	slider.style.left = e.clientX + "px";
}

const slider = document.getElementById("slider");

slider.addEventListener("pointerdown", beginSliding);
slider.addEventListener("pointerup", stopSliding);
```

---

### Write Text to Clipboard (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/clipboard

This snippet demonstrates how to copy plain text to the system clipboard using the `navigator.clipboard.setContent()` method. It requires the `readAndWrite` clipboard permission in the `manifest.json` file. The function takes the text to be copied as an argument and logs success or error messages.

```javascript
// Copy formatted text to the clipboard ✂️
async function copyToClipboard(text) {
	try {
		await navigator.clipboard.setContent({
			"text/plain": text,
		});
		console.log("✅ Text copied to clipboard");
	} catch (err) {
		console.error("❌ Failed to copy to clipboard:", err);
	}
}

// Example usage
copyToClipboard("Welcome to UXP for Premiere!");
```

```json
{
	// ...
	"requiredPermissions": {
		"clipboard": "readAndWrite"
	}
	// ...
}
```

---

### WebView Limitations

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

This section outlines the known limitations of the HTMLWebViewElement in UXP, including content loading restrictions, link behavior, and domain wildcard usage.

```APIDOC
## WebView Limitations

### Description

This section details important limitations to be aware of when using the `HTMLWebViewElement` in UXP plugins, covering content loading, link handling, and domain configuration.

### Method

N/A

### Endpoint

N/A

### Parameters

N/A

### Request Example

N/A

### Response

N/A

#### Success Response (200)

N/A

#### Response Example

N/A

### Limitations:

1.  **Local Content Loading**: UXP v7.4.3 and below only support remote content. Loading local HTML files requires UXP v8.0 or higher.
2.  **Link Behavior**: Links within a UXP WebView do not open in new windows. Actions like `<a href="..." target="_blank">` or `alert()` that would typically open new windows in a browser are not supported.
3.  **Domain Wildcards**: The `requiredPermissions.webview.domains` manifest attribute does not support top-level wildcards (`*`) for domain names. Examples of unsupported formats include `"https://www.*"` or `"https://www.adobe.*"`.
```

---

### Create an H1 Heading Element in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-html/Hierarchy/h1

Demonstrates how to create a standard HTML H1 heading element within a UXP application. This element is not theme-aware by default. It requires no external dependencies for basic rendering.

```html
<h1>Hello, world!</h1>
```

---

### Handle sp-menu change event (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-menu

Shows how to attach an event listener to an `sp-menu` component to detect changes, such as item selection. The listener logs the index of the selected item to the console. Requires a menu element with the class 'yourMenu'.

```javascript
document.querySelector(".yourMenu").addEventListener("change", (evt) => {
	console.log(`Selected item: ${evt.target.selectedIndex}`);
});
```

---

### Focus and Blur

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLButtonElement

Methods to manage the focus state of an element.

````APIDOC
## focus()

### Description
Sets the focus on the element.

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
// element.focus();
````

### Response

#### Success Response (200)

[No explicit return value, behavior is side-effectual]

#### Response Example

[None]

## blur()

### Description

Removes focus from the element.

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
// element.blur();
```

### Response

#### Success Response (200)

[No explicit return value, behavior is side-effectual]

#### Response Example

[None]

````

--------------------------------

### Query Selector for First Element

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLStyleElement

Returns the first `Element` within the document that matches a specified CSS selector. If no matching element is found, it returns `null`.

```javascript
const firstDiv = document.querySelector('div.container');
if (firstDiv) {
  firstDiv.classList.add('highlight');
}
````

---

### Focus and Blur Management in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLProgressElement

Provides methods to manage the focus state of an element. The `focus()` method brings focus to the element, while the `blur()` method removes focus from it.

```javascript
element.focus();
element.blur();
```

---

### shell.openExternal(url, developerText)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/shell/Shell

Opens a URL in the dedicated system application for the scheme. The 'file' scheme is not allowed; use `openPath` for file operations.

````APIDOC
## GET /shell/openExternal

### Description
Opens the URL in the dedicated system applications for the scheme. Note: 'file' scheme is not allowed for `openExternal`. Use `openPath` for those cases.

### Method
GET

### Endpoint
`/shell/openExternal`

### Parameters
#### Query Parameters
- **url** (string) - Required - String representing the URL to open.
- **developerText** (string) - Optional - Information from the plugin developer to be displayed on the user consent dialog. Message should be localized in the current host UI locale.

### Request Example
```javascript
shell.openExternal("https://www.adobe.com/");
shell.openExternal("https://www.adobe.com/", "develop message for the user consent dialog");

// Example with specific schemes
shell.openExternal("maps://?address=345+Park+Ave+San+Jose"); // for MacOS
shell.openExternal("bingmaps://?q=345+Park+Ave+San+Jose,+95110"); // for Windows
````

### Response

#### Success Response (200)

- **message** (string) - Resolves with an empty string if the operation succeeded.

#### Error Response (400)

- **message** (string) - String containing the error message if the operation failed.

````

--------------------------------

### Render Text Label with sp-label

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/Typography/sp-label

Demonstrates how to render a basic text label using the sp-label component. This is a fundamental UI element for displaying text.

```html
<sp-label>This is a label</sp-label>
````

---

### Element - before

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLImageElement

Inserts nodes just inside the boundary of the parent element, before the first child. This method accepts multiple nodes as arguments.

````APIDOC
## before(...nodes)

### Description
Inserts nodes just inside the boundary of the parent element, before the first child.

### Method
POST

### Endpoint
`/websites/developer_adobe_premiere-pro_uxp/before`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **...nodes** (Array<Node>) - Required - A list of nodes to insert before the first child.

### Request Example
```json
{
  "...nodes": [
    {
      "tagName": "P",
      "textContent": "First node"
    },
    {
      "tagName": "SPAN",
      "textContent": "Second node"
    }
  ]
}
````

### Response

#### Success Response (200)

(No specific return value described)

#### Response Example

```json
{}
```

````

--------------------------------

### Configure Network Access Permissions in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

This JSON configuration allows UXP plugins to specify network access permissions. It can restrict access to specific domains or grant access to all domains. The 'domains' property can be an array of URLs or the string 'all'.

```json
{
  "domains": [
    "https://example.com",
    "https://*.adobe.com/",
    "wss://*.myplugin.com"
  ]
}
````

```json
{
	"domains": "all"
}
```

---

### Pointer Capture

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHeadElement

Methods for managing pointer capture on elements, allowing for continuous tracking of a pointer.

````APIDOC
## setPointerCapture(pointerId)

### Description
Sets pointer capture for the element. This implementation does not dispatch the `gotpointercapture` event on the element. Throws `DOMException` if the element is not connected to the DOM.

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
// Assuming 'e' is a PointerEvent object from a pointerdown event
// element.setPointerCapture(e.pointerId);
````

### Response

#### Success Response (200)

N/A

#### Response Example

N/A

## releasePointerCapture(pointerId)

### Description

Releases pointer capture for the element. This implementation does not dispatch the `lostpointercapture` event on the element.

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
// Assuming 'e' is a PointerEvent object from a pointerup event
// element.releasePointerCapture(e.pointerId);
```

### Response

#### Success Response (200)

N/A

#### Response Example

N/A

````

--------------------------------

### HTMLProgressElement Properties and Methods

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLProgressElement

This section details the properties and methods available for the HTMLProgressElement interface, used for creating and managing progress bars.

```APIDOC
## HTMLProgressElement

### Description
Represents an element that supports progress indication for an operation.

### Method
N/A (This is an interface definition)

### Endpoint
N/A

### Properties
#### max : `number`
Maximum value for the progress bar.

#### value : `number`
Value of the progress bar.

#### position : `number`
Read only. Completion value of the progress bar.

#### dataset
Access to all the custom data attributes (data-*) set.
See: HTMLElement - dataset

#### innerText : `string`

#### lang : `string`
Base language of an element's attribute values and text content.
See: HTMLElement - lang

#### dir : `string`
The text writing directionality of the content of the current element limited to only known values.
See: HTMLElement - dir
Since: v7.1

#### hidden : `boolean` | `string`
Indicates the browser should not render the contents of the element. Note: "until-found" is not supported.
See: HTMLElement - hidden, Spec - `hidden` attribute

#### nodeName : `string`
Read only.

#### localName : `string`
Read only. A string representing the local part of the qualified name of the element.
See: https://developer.mozilla.org/en-US/docs/Web/API/Element/localName

#### tagName : `string`
Read only. A string indicating the element's tag name.
See: https://developer.mozilla.org/en-US/docs/Web/API/Element/tagName

#### nodeType : `number`
Read only.

#### namespaceURI : `string`
Read only. Returns the namespace URI of the element, or null if the element is not in a namespace.
See: https://developer.mozilla.org/en-US/docs/Web/API/Element/namespaceURI

#### id : `string`
Returns the property of the `Element` interface represents the element's identifier, reflecting the id global attribute.
See: https://developer.mozilla.org/en-US/docs/Web/API/Element/id

#### tabIndex : `number`

#### className : `string`

#### attributes : `NamedNodeMap`
Read only.

#### style : `Style`
Read only.

#### clientLeft : `number`
Read only.

#### clientTop : `number`
Read only.

#### clientWidth : `number`
Read only.

#### clientHeight : `number`
Read only.

#### offsetParent : `Element`
Read only.

#### offsetLeft : `number`
Read only.

#### offsetTop : `number`
Read only.

#### offsetWidth : `number`
Read only.

#### offsetHeight : `number`
Read only.

#### scrollLeft : `number`

#### scrollTop : `number`

#### scrollWidth : `number`
Read only.

#### scrollHeight : `number`
Read only.

#### autofocus : `boolean`
Indicates if the element will focus automatically when it is loaded.

#### uxpContainer : `number`
Read only.

#### shadowRoot : `ShadowRoot`
Read only. [ This feature is behind a feature flag. You must turn on `enableSWCSupport` in the featureFlags section of plugin manifest to use the same ]
Returns the open shadow root that is hosted by the element, or null if no open shadow root is present.
See: Element - shadowRoot

#### disabled : `boolean`

#### innerHTML
Read only.

#### outerHTML : `string`

#### slot : `string`
[ This feature is behind a feature flag. You must turn on `enableSWCSupport` in the featureFlags section of plugin manifest to use the same ]
See: Element - slot

#### assignedSlot : `HTMLSlotElement`
Read only. [ This feature is behind a feature flag. You must turn on `enableSWCSupport` in the featureFlags section of plugin manifest to use the same ]
See: Element - assignedSlot

#### contentEditable
Read only.

#### isConnected : `boolean`
Read only.

#### parentNode : `Node`
Read only.

#### parentElement : `Element`
Read only.

#### firstChild : `Node`
Read only.

#### lastChild : `Node`
Read only.

#### previousSibling : `Node`
Read only.

#### nextSibling : `Node`
Read only.

#### firstElementChild : `Node`
Read only.

#### lastElementChild : `Node`
Read only.

#### previousElementSibling : `Node`
Read only.

#### nextElementSibling : `Node`
Read only.

#### textContent : `string`

#### childNodes : `NodeList`
Read only.

#### children : `HTMLCollection`
Read only.

#### ownerDocument
Read only.

### Methods
#### append(...nodes)
Inserts a set of Node objects or string objects after the last child of the Element.
See: https://developer.mozilla.org/en-US/docs/Web/API/Element/append
Since: v8.0
Parameters:
- **...nodes** (Array<Node>)

#### prepend(...nodes)
Inserts a set of Node objects or string objects before the first child of the Element.
See: https://developer.mozilla.org/en-US/docs/Web/API/Element/prepend
Since: v8.0
Parameters:
- **...nodes** (Array<Node>)

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/HTMLProgressElement
- HTMLElement
````

---

### Import SWC Button Component

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/swc

Import the Spectrum Web Component's button JavaScript file into your project to make it available for use. This is typically done at the top of your component or page file.

```javascript
import "@swc-uxp-wrappers/button/sp-button.js";
```

---

### Query Selector All for Elements (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLScriptElement

Returns a non-live `NodeList` representing a list of the document's elements that match the specified group of selectors. If no matches are found, an empty `NodeList` is returned.

```javascript
function querySelectorAll(selector) {
	// Implementation details...
}
```

---

### Load Media in UXP Video Element - JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLVideoElement

Loads media into a video element and listens for 'loadedmetadata' and 'loadeddata' events. This method resets the media to its initial state and prepares it for playback.

```javascript
<video id="sampleVideo"  src="https://images-tv.adobe.com/mpcv3/b6a5d5f7-5a6c-4bd6-9ee9-ddb6c9c779b3_1564010305.854x480at800_h264.mp4" preload="none">
</video>
<script>
let vid = document.getElementById("sampleVideo");
vid.load();
vid.addEventListener("loadedmetadata", (ev) => {
    console.log("Event - loadedmetadata");
});
vid.addEventListener("loadeddata", (ev) => {
    console.log("Event - loadeddata");
});
</script>
```

---

### Read File Synchronously - UXP FSAPI

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/fs

Reads data from a specified file path synchronously. Supports binary or encoded text formats via options. Returns the file content directly. This method blocks the main thread until the file operation is complete.

```javascript
const data = fs.readFileSync("plugin-data:/binaryFile.obj");
```

```javascript
const text = fs.readFileSync("plugin-data:/textFile.txt", { encoding: "utf-8" });
```

---

### insertBefore

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLStyleElement

Inserts a node before a specified existing node in the parent's child list.

````APIDOC
## insertBefore(child, before)

### Description
Inserts a node before a specified existing node in the parent's child list.

### Method
POST (as it modifies the DOM)

### Endpoint
Not applicable for this method as it's called directly on an element instance.

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Assuming 'parentElement', 'newChildNode', and 'referenceNode' are DOM Nodes
parentElement.insertBefore(newChildNode, referenceNode);
````

### Response

#### Success Response (200)

- **Node** - The inserted node.

#### Response Example

```json
// Represents the inserted Node object
{
	"nodeType": 1,
	"nodeName": "DIV"
}
```

````

--------------------------------

### Query Selector All for Matching Elements

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLStyleElement

Returns a static `NodeList` representing a list of all elements within the document that match the specified CSS selectors. If no matches are found, it returns an empty `NodeList`.

```javascript
const allButtons = document.querySelectorAll('button.primary');
allButtons.forEach(button => {
  button.style.backgroundColor = 'green';
});
````

---

### Copy and Paste Text Transformation (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/clipboard

This snippet demonstrates a combined clipboard operation: reading text, transforming it (to uppercase in this case), and then writing the transformed text back to the clipboard. It requires `readAndWrite` permission and uses both `navigator.clipboard.getContent()` and `navigator.clipboard.setContent()`. Error handling is included.

```javascript
// Transform clipboard text to uppercase
async function transformClipboardText() {
	try {
		// Read current clipboard content
		const data = await navigator.clipboard.getContent();

		if (data["text/plain"]) {
			const originalText = data["text/plain"];
			const transformedText = originalText.toUpperCase();

			// Write the transformed text back
			await navigator.clipboard.setContent({
				"text/plain": transformedText,
			});

			console.log(`✅ Transformed: "${originalText}" → "${transformedText}"`);
		} else {
			console.log("⚠️ No text found on clipboard");
		}
	} catch (err) {
		console.error("Failed to transform clipboard text:", err);
	}
}

// Example usage
transformClipboardText();
```

---

### getPluginFolder

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider

Returns a read-only folder containing plugin-related packaged assets.

```APIDOC
## GET /getPluginFolder

### Description
Returns a plugin's folder – this folder and everything within it are read only. This contains all the Plugin related packaged assets.

### Method
GET

### Endpoint
`/getPluginFolder`

### Returns
`Promise<Folder>`
```

---

### CSS: Apply Styles Based on Theme with prefers-color-scheme

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Media%20Queries/prefers-color-scheme

This CSS snippet demonstrates how to use the @media (prefers-color-scheme) query to apply different styles based on the host application's theme. It defines default variables for dark themes and overrides them for light themes. This is useful for ensuring your plugin's UI matches the user's chosen color scheme.

```css
:root {
	--primary-color: #e8e8e8; /* default colors are for dark themes */
}
@media (prefers-color-scheme: lightest), (prefers-color-scheme: light) {
	:root {
		--primary-color: #181818; /* override for light themes */
	}
}
```

---

### sort - Sort XMP Contents

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPMeta

Sorts the XMP contents alphabetically by namespaces, properties, struct fields, and array items.

````APIDOC
## sort()

### Description
Sorts the XMP contents alphabetically. At the top level, namespaces are sorted by their prefixes. Within a namespace, top-level properties are sorted by name. Within a struct, fields are sorted by their qualified name (e.g., `prefix:localName`). Unordered arrays of simple items are sorted by value. Language alternative arrays (`alt`) are sorted by their `xml:lang` qualifiers, with the `x-default` item placed first.

### Method
`sort`

### Parameters
None

### Request Example
```javascript
XMPMetaObj.sort()
````

### Response

#### Success Response (200)

This method does not return a specific value upon success; it modifies the XMP object in place.

#### Response Example

N/A

````

--------------------------------

### Plugin to WebView Communication (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

Demonstrates how a UXP plugin can send messages to its embedded WebView and receive responses. It utilizes `HTMLWebViewElement.postMessage` to send and `window.addEventListener('message', ...)` to receive. Messages are stringified and parsed using JSON.

```javascript
// Send message from plugin to WebView
let webViewDisplay = document.getElementById("webviewSample");
webViewDisplay.postMessage("PluginMessage1");


// Plugin receives message from WebView via "message" event.
window.addEventListener("message", (e) => {
  console.log(`Message from WebView(Origin:${e.origin}): ${e.data}\n`);


  if (e.data === "webviewmessage1") {
    webViewDisplay.postMessage("Thanks, Message1 recieved successfully");
  }
});
````

---

### Element Focus and Blur

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLSelectElement

Methods for managing element focus.

```APIDOC
## focus()

### Description
Sets the focus on the element.

### Method
`focus`
```

```APIDOC
## blur()

### Description
Removes focus from the element.

### Method
`blur`
```

---

### Perform Version-Based Feature Detection in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/host-info

This JavaScript function checks if the current UXP runtime version meets a minimum requirement (e.g., 8.1 or higher) to enable advanced features. It parses the `versions.uxp` string and returns a boolean. This is useful for conditionally enabling features and ensuring compatibility.

```javascript
const { host, versions } = require("uxp");

// Check if UXP version supports a feature 🔍
function supportsAdvancedFeatures() {
	const uxpVersion = versions.uxp;

	// Parse version string (e.g., "8.1.0" -> 8.1)
	const majorMinor = parseFloat(uxpVersion.split(".").slice(0, 2).join("."));

	// Check if UXP is version 8.1 or higher
	return majorMinor >= 8.1;
}

// Conditionally enable features based on version
function initializePlugin() {
	console.log("Initializing plugin...");

	if (supportsAdvancedFeatures()) {
		console.log("✅ Advanced features enabled (UXP 8.1+)");
		// Enable newer API usage
	} else {
		console.log("⚠️ Using legacy mode (UXP < 8.1)");
		// Provide fallback behavior
	}
}
```

---

### Element Selection with getElementsByClassName, getElementsByTagName, querySelector, querySelectorAll

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHeadElement

These methods facilitate selecting elements from the DOM. `getElementsByClassName` and `getElementsByTagName` return collections of elements matching a specific class name or tag name, respectively. `querySelector` returns the first element matching a CSS selector, while `querySelectorAll` returns all matching elements.

```javascript
// Get elements by class name
const itemsByClass = document.getElementsByClassName("my-item");

// Get elements by tag name
const divs = document.getElementsByTagName("div");

// Query the first element matching a CSS selector
const firstButton = document.querySelector("button.submit");

// Query all elements matching a CSS selector
const allParagraphs = document.querySelectorAll("p.intro");
```

---

### Element Querying and Selection

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLButtonElement

Methods for selecting elements within the DOM. `getElementsByClassName` returns a NodeList of all elements with a given class name. `getElementsByTagName` returns a NodeList of all elements with a given tag name. `querySelector` returns the first element matching a CSS selector, and `querySelectorAll` returns a NodeList of all elements matching a CSS selector.

```javascript
/**
 * Returns a NodeList of all elements in the document that have the specified class name.
 * @param {string} name - The class name to search for.
 * @returns {NodeList} A NodeList of elements.
 */
function getElementsByClassName(name) {
	// Implementation details...
	return new NodeList(); // Placeholder
}

/**
 * Returns a NodeList of all elements in the document with a given tag name.
 * @param {string} name - The tag name to search for.
 * @returns {NodeList} A NodeList of elements.
 */
function getElementsByTagName(name) {
	// Implementation details...
	return new NodeList(); // Placeholder
}

/**
 * Returns the first element in the document that matches a specified selector.
 * @param {string} selector - A DOMString representing the selector to match.
 * @returns {Element | null} The first matching element or null if no match is found.
 */
function querySelector(selector) {
	// Implementation details...
	return null; // Placeholder
}

/**
 * Returns a NodeList representing a list of the document's elements that match the specified group of selectors.
 * @param {string} selector - A DOMString representing the selector to match.
 * @returns {NodeList} A NodeList of matching elements.
 */
function querySelectorAll(selector) {
	// Implementation details...
	return new NodeList(); // Placeholder
}
```

---

### Read Binary File Content using UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/File

Shows how to read a file's content as a binary ArrayBuffer using the 'read' method with the 'format' option set to 'formats.binary'. This is useful for handling non-textual data.

```javascript
const data = await myNovel.read({ format: formats.binary });
```

---

### Node and Tree Traversal

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLTextAreaElement

Methods for checking node relationships and traversing the DOM tree.

````APIDOC
## hasChildNodes()

### Description
Checks if the element has any child nodes.

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
// const hasChildren = element.hasChildNodes();
````

### Response

#### Success Response (200)

- **boolean** - `true` if the element has child nodes, `false` otherwise.

#### Response Example

```json
true
```

````

```APIDOC
## contains(node)

### Description
Checks whether a node is a descendant of a given node, i.e. is it a child, grandchild, etc. of the given node.

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
// const isContained = parentElement.contains(childElement);
````

### Response

#### Success Response (200)

- **boolean** - `true` if the node is contained within the element, `false` otherwise.

#### Response Example

```json
true
```

````

```APIDOC
## getRootNode(options)

### Description
Returns the root of the current node's subtree.
Note: Shadow DOM and other DOM fragmentation models may alter the method's behavior.

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
// const root = element.getRootNode();
````

### Response

#### Success Response (200)

- **Node** - The root node of the subtree.

#### Response Example

```json
{
	"nodeType": 11, // DocumentFragment
	"nodeName": "#document-fragment"
}
```

````

--------------------------------

### Element Geometry API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLOptionElement

APIs for retrieving an element's dimensions and position.

```APIDOC
## getBoundingClientRect()

### Description
Returns a `DOMRect` object providing information about the size of an element and its position relative to the viewport.

### Method
GET (Implied)

### Endpoint
Element.getBoundingClientRect

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
{}
````

### Response

#### Success Response (200)

- **elementRect** (DOMRect) - An object with `top`, `right`, `bottom`, `left`, `width`, and `height` properties.

#### Response Example

```json
{
	"top": 10,
	"right": 100,
	"bottom": 50,
	"left": 20,
	"width": 80,
	"height": 40
}
```

````

--------------------------------

### Enable Alerts Feature Flag in manifest.json

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/debug

Shows the necessary configuration in the `manifest.json` file to enable dialog-based debugging features like `alert()`, `confirm()`, and `prompt()` for UXP plugins. This involves setting the `enableAlerts` flag to `true` within the `featureFlags` object.

```json
{
  "manifestVersion": 5,
  // ...
  "featureFlags": {
    "enableAlerts": true
  }
  // ...
}

````

---

### Element Query API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLDialogElement

APIs for finding elements by class name or tag name.

```APIDOC
## getElementsByClassName(name)

### Description
Returns a live `NodeList` collection of all child elements which have all of the given class names.

### Method
`getElementsByClassName`

### Parameters
#### Path Parameters
- **name** (string) - A string containing one or more class names separated by spaces.

### Returns
`NodeList`

---

## getElementsByTagName(name)

### Description
Returns a live `NodeList` collection of all elements in the document or a specific element that have the specified tag name.

### Method
`getElementsByTagName`

### Parameters
#### Path Parameters
- **name** (string) - The tag name to search for (e.g., 'div', 'p').

### Returns
`NodeList`
```

---

### Retrieve File or Folder Entry by URL

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider

Retrieves a file or folder entry corresponding to the given URL. This function is useful for accessing existing files or directories within the UXP environment. It throws an error if the file or folder does not exist at the specified URL.

```javascript
const tmpFolder = await fs.getEntryWithUrl("plugin-temp:/tmp");
const docFolder = await fs.getEntryWithUrl("file:/Users/user/Documents");
```

```javascript
const tmpFile = await fs.getEntryWithUrl("plugin-temp:/tmp/test.dat");
const docFile = await fs.getEntryWithUrl("file:/Users/user/Documents/test.txt");
```

---

### Render Theme-Aware Body Text with sp-body

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/Typography/sp-body

Demonstrates how to use the sp-body component to render body text that adapts to the current theme. This component requires no external dependencies for basic usage.

```html
<sp-body>This is some body text</sp-body>
```

---

### FolderItem Static Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/folderitem

Static methods available on the FolderItem class, such as casting.

```APIDOC
## FolderItem Static Methods

### Description
Static methods available on the FolderItem class.

### Method: cast
#### Description
Cast ProjectItem into FolderItem.
#### Parameters
- **projectItem** (_ProjectItem_) - Description: -
```

---

### Element Focus and Blur Methods: focus, blur

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLMenuItemElement

Provides methods to programmatically set the focus or blur state of an element. `focus()` will move the focus to the element, and `blur()` will remove focus from it. These are essential for controlling user interaction and keyboard navigation.

```javascript
/**
 * Programmatically sets focus to the element.
 */
function focus() {
	// Implementation details not provided in source
}

/**
 * Removes focus from the element.
 */
function blur() {
	// Implementation details not provided in source
}
```

---

### Configure JavaScript IntelliSense with JSDoc in Premiere Pro UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/typescript-support

This configuration enables JavaScript IntelliSense using JSDoc comments. It requires the `types.d.ts` file and a `jsconfig.json` file in the plugin's root directory. This approach provides editor warnings for type errors and works without a build step.

```json
{
	"compilerOptions": {
		"checkJs": true,
		"target": "ES2020",
		"lib": ["ES2020", "DOM"]
	},
	"include": ["*.js"],
	"exclude": ["node_modules", "dist"]
}
```

---

### Open Multiple Modals with Independent Buttons (HTML)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-modal-dialogs

This HTML snippet demonstrates how to structure a document with multiple `<dialog>` elements, each intended to be opened by a separate button. It includes basic Adobe Spectrum components for UI elements. The `id` attribute is crucial for referencing the dialogs in JavaScript.

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
		<sp-heading>Open Modal Dialog</sp-heading>
		<sp-button-group>
			<sp-button id="openFirstDialogBtn">Open First Dialog</sp-button>
			<sp-button id="openSecondDialogBtn">Open Second Dialog</sp-button>
		</sp-button-group>

		<!-- first modal -->
		<dialog id="modal1">
			<sp-heading>👋 Hello Modal Dialog!</sp-heading>
			<sp-divider size="L"></sp-divider>
			<sp-body>Modal body content 1</sp-body>
		</dialog>

		<!-- second modal -->
		<dialog id="modal2">
			<sp-heading>👋 Hello Another Modal Dialog!</sp-heading>
			<sp-divider size="L"></sp-divider>
			<sp-body>Modal body content 2</sp-body>
		</dialog>
	</body>
</html>
```

---

### EventTarget Methods

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLMenuElement

Standard methods for adding, removing, and dispatching events on an element, inherited from EventTarget.

````APIDOC
## addEventListener(eventName, callback, options)

### Description
Attaches an event listener to the element.

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
element.addEventListener('click', handleClick, { capture: true });
````

### Response

#### Success Response (200)

N/A (Method does not return a value)

#### Response Example

N/A

---

## removeEventListener(eventName, callback, options)

### Description

Removes an event listener from the element.

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
// Assuming 'element' is a reference to an HTML element and 'handleClick' is the same function used in addEventListener
element.removeEventListener("click", handleClick, { capture: true });
```

### Response

#### Success Response (200)

N/A (Method does not return a value)

#### Response Example

N/A

---

## dispatchEvent(event)

### Description

Dispatches an event to the element.

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
// Assuming 'element' is a reference to an HTML element and 'customEvent' is a valid Event object
element.dispatchEvent(customEvent);
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

### Write File Data Asynchronously - JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/fs

Writes data in chunks to a file specified by its file descriptor. It takes data from a buffer and writes it to the file. Supports specifying the buffer, offset, length, and position for writing. Returns the number of bytes written. If no callback is provided, it returns a Promise.

```javascript
const fd = await fs.open("plugin-data:/fileToWrite.txt", "w+");
const data = "It was a dark and stormy night.\n";
const srcBuffer = new Uint8Array(data.length);
for (let i = 0; i < data.length; i++) {
 srcBuffer[i] = data.charCodeAt(i);
}
const { bytesWritten } = await fs.write(fd, srcBuffer.buffer, 0, srcBuffer.length, 0);
await fs.close(fd);
````

---

### Safe Fetch with Timeout and Error Handling

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/network

A robust asynchronous function for making network requests using the fetch API. It includes a configurable timeout, automatically aborts requests that exceed it, checks for HTTP errors using `res.ok`, and catches network exceptions. Logs detailed error information. Requires appropriate network permissions.

```javascript
async function safeFetch(url, options = {}, timeoutMs = 8000) {
	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), timeoutMs);

	try {
		const res = await fetch(url, { ...options, signal: controller.signal });
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		return await res.json();
	} catch (err) {
		console.error("Network request failed:", err);
		throw err;
	} finally {
		clearTimeout(timeout);
	}
}
```

---

### Request Properties

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/Request

Explains the read-only properties of the Request object, including its body, bodyUsed status, credentials, headers, method, and URL.

````APIDOC
## Request Properties

### Description
Read-only properties of the `Request` object providing information about the request.

### Properties
- **body** (`ReadableStream` | `null`)
  - Description: ReadableStream object with the body contents that have been added to the request or null if body is null.
  - See: Request - body
- **bodyUsed** (`boolean`)
  - Description: Indicates whether the request body has been read yet.
  - See: Request - bodyUsed
- **credentials** (`string`)
  - Description: Indicates whether the user agent should send or receive cookies. Possible values are: "omit" (Never send or receive cookies), "include" (Always send cookies).
  - See: Request - credentials
- **headers** (`Headers`)
  - Description: Headers object associated with the request.
  - See: Request - headers
- **method** (`string`)
  - Description: Request's method. Possible values are "GET", "POST", "HEAD", "PUT", "DELETE" and "OPTIONS".
  - See: Request - method
- **url** (`string`)
  - Description: URL of the request.
  - See: Request - url

### Request Example
```javascript
// Example usage is not provided in the source text.
````

### Response Example

```json
{
	"example": "property details"
}
```

````

--------------------------------

### Implement Theme Awareness in UXP Plugins

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/css-styling

Shows how to make a UXP plugin's UI adapt to Premiere Pro's current theme (Light, Dark, or Darkest). It utilizes the `document.theme` API to detect theme changes and update element styles accordingly, ensuring a consistent user experience across different themes.

```javascript
function updateTheme(theme) {
  panelBody = document.getElementById("plugin-body");
  panelHeading = document.getElementById("plugin-heading");

  // Change styles based on the new theme on the fly
  if(theme.includes("dark")) {
    panelBody.style.color = "#fff";
    panelHeading.style.color = "#fff";
  } else {
    panelBody.style.color = "#000";
    panelHeading.style.color = "#000";
  }
}

// Listen for theme changes
document.theme.onUpdated.addListener((theme) => {
    updateTheme(theme);
})

// Apply initial theme on load
const currentTheme = document.theme.getCurrent();
updateTheme(currentTheme);
````

---

### Query Selector for First Element (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLabelElement

Returns the first element within the document that matches a specified CSS selector. If no match is found, it returns null.

```javascript
/**
 * Returns the first element within the document that matches the specified CSS selector.
 *
 * @param {string} selector - The CSS selector to match.
 * @returns {Element|null} The first matching element, or null if none found.
 */
function querySelector(selector) {
	// Implementation details for query selector
}
```

---

### Element Matching and Traversal API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLOptionElement

APIs for checking if an element matches a selector and finding the closest ancestor element that matches a selector.

````APIDOC
## closest(selectorString)

### Description
Returns the closest ancestor element (or the element itself) that matches the specified CSS selector.

### Method
GET (Implied)

### Endpoint
Element.closest

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
  "selectorString": ".my-class"
}
````

### Response

#### Success Response (200)

- **closestElement** (Element) - The matching ancestor element, or null if no match is found.

#### Response Example

```json
{
	"closestElement": "<div class='parent'>...</div>"
}
```

## matches(selectorString)

### Description

Determines whether an element matches a specified CSS selector.

### Method

GET (Implied)

### Endpoint

Element.matches

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
	"selectorString": "div#my-id"
}
```

### Response

#### Success Response (200)

- **isMatch** (boolean) - True if the element matches the selector, false otherwise.

#### Response Example

```json
{
	"isMatch": true
}
```

````

--------------------------------

### Query Selector for Elements (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLTextAreaElement

Methods to select elements based on CSS selectors. `querySelector` returns the first matching element, while `querySelectorAll` returns all matching elements in a NodeList.

```javascript
/**
 * Returns the first element that is a descendant of the element that matches the specified selector.
 * @param {string} selector - The CSS selector to match.
 * @returns {Element|null} The first matching element, or null if no match is found.
 */
Element.prototype.querySelector(selector);
````

```javascript
/**
 * Returns a NodeList representing a list of the element's descendants that match the specified selectors.
 * @param {string} selector - The CSS selector to match.
 * @returns {NodeList} A NodeList of all matching elements.
 */
Element.prototype.querySelectorAll(selector);
```

---

### Set and Release Pointer Capture for Dragging Elements (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLOptionElement

Demonstrates how to implement drag-and-drop functionality using `setPointerCapture` and `releasePointerCapture`. This allows an element to continuously track pointer movements even if the pointer leaves its bounds. It requires event listeners for pointer down, move, and up events. The `slide` function updates the element's position based on pointer coordinates.

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
```

---

### Element Query and Geometry

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLTextAreaElement

Methods for querying element properties and dimensions.

````APIDOC
## getBoundingClientRect()

### Description
Returns the size of an element and its position relative to the viewport.

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
// const rect = element.getBoundingClientRect();
````

### Response

#### Success Response (200)

- **\* (DOMRect)** - An object containing `top`, `right`, `bottom`, `left`, `width`, and `height` properties.

#### Response Example

```json
{
	"top": 10,
	"right": 150,
	"bottom": 110,
	"left": 50,
	"width": 100,
	"height": 100
}
```

````

```APIDOC
## closest(selectorString)

### Description
Returns the first ancestor of the element that matches the given selector string.

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
// const parentDiv = element.closest('.container');
````

### Response

#### Success Response (200)

- **Element** - The closest ancestor element, or `null` if no match is found.

#### Response Example

```json
{
	"tagName": "DIV",
	"id": "container"
}
```

````

```APIDOC
## matches(selectorString)

### Description
Tests whether an element matches a CSS selector.

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
// const isVisible = element.matches('.visible');
````

### Response

#### Success Response (200)

- **boolean** - `true` if the element matches the selector, `false` otherwise.

#### Response Example

```json
true
```

````

--------------------------------

### TransitionFactory - getVideoTransitionMatchNames

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/transitionfactory

Returns a promise that resolves with an array of available video transition match names. This is useful for understanding which transitions can be created using createVideoTransition.

```APIDOC
## GET /websites/developer_adobe_premiere-pro_uxp/TransitionFactory/getVideoTransitionMatchNames

### Description
Return a promise which will be fullfilled with an array of video transition matchnames.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/TransitionFactory/getVideoTransitionMatchNames

### Parameters
No parameters required for this endpoint.

### Request Example
```json
{}
````

### Response

#### Success Response (200)

- **matchNames** (string[]) - An array of strings, where each string is a match name for a video transition.

````

--------------------------------

### Basic sp-button Rendering

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-button

Renders a basic button. This component is part of the Spectrum Web Components library and is available since UXP v4.1.

```html
<sp-button>Vectorize</sp-button>
````

---

### Event Target APIs

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLDialogElement

APIs for managing event listeners on an element.

````APIDOC
## addEventListener(eventName, callback, options)

### Description
Attaches an event listener to the element.

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
element.addEventListener('click', handleClick);
````

### Response

#### Success Response (200)

No return value. The event listener is attached.

### Parameters

- **eventName** (\*) - The name of the event to listen for.
- **callback** (\*) - The function to be called when the event is triggered.
- **options** (boolean | Object) - A boolean value denoting capture value or options object. Currently supports only capture in options object ({ capture: bool_value }).

### See

- EventTarget - addEventListener

## removeEventListener(eventName, callback, options)

### Description

Removes an event listener from the element.

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
element.removeEventListener("click", handleClick);
```

### Response

#### Success Response (200)

No return value. The event listener is removed.

### Parameters

- **eventName** (\*) - The name of the event.
- **callback** (\*) - The event handler function that was added using addEventListener().
- **options** (boolean | Object) - A boolean value denoting capture value or options object. Currently supports only capture in options object ({ capture: bool_value }).

### See

- EventTarget - removeEventListener

## dispatchEvent(event)

### Description

Dispatches an `Event` into the event flow of the element.

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
const myEvent = new Event("customEvent");
element.dispatchEvent(myEvent);
```

### Response

#### Success Response (200)

- **boolean** (boolean) - `true` if the event was dispatched successfully, `false` otherwise.

### Parameters

- **event** (\*) - The event to dispatch.

````

--------------------------------

### HTML for Modal with Close Buttons and Return Values

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-modal-dialogs

This HTML structure defines a modal dialog designed to return values upon closure. It includes 'OK' and 'Cancel' buttons, each with a unique ID, to trigger the closing mechanism. The `dialog` element itself is referenced in JavaScript to manage its state and return data.

```html
<!DOCTYPE html>
<html>
<head>
  <script src="main.js"></script>
  <link rel="stylesheet" href="style.css" />
</head>

<body>
  <sp-heading>Open Modal Dialog</sp-heading>
  <sp-button id="openDialogBtn">Click</sp-button>


  <dialog>
    <sp-heading>👋 Hello Modal Dialog!</sp-heading>
    <sp-divider size="L"></sp-divider>
    <sp-body>Modal body content</sp-body>
    <sp-button-group>
      <sp-button id="closeDialogBtn">OK</sp-button>
      <sp-button id="cancelDialogBtn">Cancel</sp-button>
    </sp-button-group>
  </dialog>


</body>
</html>
````

---

### HTMLVideoElement: Basic HTML Structure

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLVideoElement

Provides a basic HTML structure for a video element with 'preload' and 'src' attributes. The 'preload' attribute controls how much data is loaded when the plugin loads, affecting initial performance.

```html
<video
	src="https://images-tv.adobe.com/mpcv3/b6a5d5f7-5a6c-4bd6-9ee9-ddb6c9c779b3_1564010305.854x480at800_h264.mp4"
	preload="metadata"
></video>
```

---

### WebView to Plugin Communication (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

Illustrates how content within a WebView can send messages to the parent UXP plugin and receive messages from it. It uses `window.uxpHost.postMessage` for sending and `window.addEventListener('message', ...)` for receiving. Message data is expected to be JSON-parsable.

```javascript
// WebView sends message to Plugin
window.uxpHost.postMessage("webviewmessage1");

// WebView receives messages from Plugin
window.addEventListener("message", (e) => {
	// (e) from Plugin
	// e.origin would be 'plugin id'
	// e.source would be 'window.uxpHost'
	// e.data is 'JSON.parse(JSON.stringify("PluginMessage1"))' which is "PluginMessage1"
	if (e.data === "PluginMessage1") {
		console.log(e.data);
	}
});
```

---

### Create a Folder Entry in a Folder (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/Folder

Demonstrates how to create a new subfolder within an existing Folder instance. This method returns the newly created Folder entry object.

```javascript
const myCollectionsFolder = await aFolder.createFolder("collections");
```

---

### Heading Component Sizes (HTML)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/Typography/sp-heading

Demonstrates the different available sizes for the sp-heading component, ranging from XXXL down to XXS. Each size can be applied using the 'size' attribute.

```html
<sp-heading size="XXXL">Heading XXXL</sp-heading>
<sp-heading size="XXL">Heading XXL</sp-heading>
<sp-heading size="XL">Heading XL</sp-heading>
<sp-heading size="L">Heading L</sp-heading>
<sp-heading size="M">Heading M</sp-heading>
<sp-heading size="S">Heading S</sp-heading>
<sp-heading size="XS">Heading XS</sp-heading>
<sp-heading size="XXS">Heading XXS</sp-heading>
```

---

### DOM Insertion API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLOptionElement

APIs for inserting HTML, elements, or text adjacent to an existing element.

````APIDOC
## insertAdjacentHTML(position, value)

### Description
Inserts an HTML string into the DOM adjacent to the element.

### Method
POST (Implied)

### Endpoint
Element.insertAdjacentHTML

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **position** (*)
- **value** (string) - The HTML string to insert.

### Request Example
```json
{
  "position": "afterend",
  "value": "<p>New paragraph</p>"
}
````

### Response

#### Success Response (200)

No specific return value documented, operation is side-effect based.

#### Response Example

```json
{}
```

## insertAdjacentElement(position, node)

### Description

Inserts a node into the DOM adjacent to the element.

### Method

POST (Implied)

### Endpoint

Element.insertAdjacentElement

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

- **position** (\*)
- **node** (\*) - The `Node` to insert.

### Request Example

```json
{
	"position": "beforebegin",
	"node": "<div id='new-element'></div>"
}
```

### Response

#### Success Response (200)

- **insertedNode** (Node) - The inserted node.

#### Response Example

```json
{
	"insertedNode": "<div id='new-element'></div>"
}
```

## insertAdjacentText(position, text)

### Description

Inserts a text node into the DOM adjacent to the element.

### Method

POST (Implied)

### Endpoint

Element.insertAdjacentText

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

- **position** (\*)
- **text** (\*) - The text content to insert.

### Request Example

```json
{
	"position": "afterbegin",
	"text": "Some text"
}
```

### Response

#### Success Response (200)

No specific return value documented, operation is side-effect based.

#### Response Example

```json
{}
```

````

--------------------------------

### size() Method

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Streams/CountQueuingStrategy

The size() method of a CountQueuingStrategy object, which returns the size of a chunk, always 1.

```APIDOC
## size()

### Description
Returns the size of the chunk, which is always 1 for CountQueuingStrategy.

### Method
`strategy.size()`

### Endpoint
N/A (Method call)

### Parameters
None

### Request Example
```javascript
const strategy = new CountQueuingStrategy({ highWaterMark: 10 });
const chunkSize = strategy.size();
console.log(chunkSize); // Output: 1
````

### Response

#### Success Response (200)

Returns the chunk size as a number (always 1).

#### Response Example

```json
{
	"chunkSize": 1
}
```

````

--------------------------------

### Element Selection by Class and Tag (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLinkElement

Methods for selecting multiple elements based on their class name or tag name. Returns a NodeList of matching elements.

```javascript
/**
 * Returns a NodeList of all elements in the document that have the specified class name.
 * @param {string} name - The class name to search for.
 * @returns {NodeList} A NodeList containing all matching elements.
 */
function getElementsByClassName(name) {
  // Implementation details...
}

/**
 * Returns a NodeList of all elements in the document that have the specified tag name.
 * @param {string} name - The tag name to search for.
 * @returns {NodeList} A NodeList containing all matching elements.
 */
function getElementsByTagName(name) {
  // Implementation details...
}

````

---

### Display a Modal Dialog using HTML and UXP JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-modal-dialogs

This snippet demonstrates how to display a modal dialog in UXP. It involves an HTML structure with a button to trigger the dialog and a hidden dialog element. The JavaScript uses `uxpShowModal()` to present the dialog with specified properties like title, resize behavior, and size. This is typically used within a Premiere Pro plugin's panel.

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
		<!-- Panel content -->
		<sp-heading>Open Modal Dialog</sp-heading>
		<sp-button id="openDialogBtn">Click</sp-button>

		<!-- Modal dialog content (hidden by default) -->
		<dialog>
			<sp-heading>👋 Hello Modal Dialog!</sp-heading>
			<sp-divider size="L"></sp-divider>
			<sp-body>Modal body content</sp-body>
		</dialog>
	</body>
</html>
```

```javascript
const openDialogBtn = document.getElementById("openDialogBtn");
openDialogBtn.addEventListener("click", () => {
	const dialog = document.querySelector("dialog");
	dialog.uxpShowModal({
		title: "Demo Modal Dialog",
		resize: "none",
		size: { width: 300, height: 300 },
	});
});
```

---

### Element Focus and Blur Methods

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLButtonElement

Provides basic methods to manage the focus state of an element. `focus()` sets the element as the current focus, while `blur()` removes focus from the element.

```javascript
/**
 * Sets the focus on the element.
 */
function focus() {
	// Implementation details...
}

/**
 * Removes focus from the element.
 */
function blur() {
	// Implementation details...
}
```

---

### Node Manipulation

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLVideoElement

This section details methods for manipulating nodes within the UXP environment, including replacing, appending, and checking for containment.

````APIDOC
## after(...nodes)

### Description
Appends nodes after the current node.

### Method
*after*

### Endpoint
N/A (Method within a class)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **...nodes** (Array<Node>) - The nodes to append.

### Request Example
```javascript
currentNode.after(newNode1, newNode2);
````

### Response

#### Success Response (200)

None (This is a method call).

#### Response Example

None

## replaceWith(...nodes)

### Description

Replaces the current node with the provided nodes.

### Method

_replaceWith_

### Endpoint

N/A (Method within a class)

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

- **...nodes** (Array<Node>) - The nodes to replace with.

### Request Example

```javascript
currentNode.replaceWith(newNode);
```

### Response

#### Success Response (200)

None (This is a method call).

#### Response Example

None

## contains(node)

### Description

Checks if the current node contains the specified child node.

### Method

_contains_

### Endpoint

N/A (Method within a class)

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

- **node** (Node) - The node to check for containment.

### Request Example

```javascript
if (parentNode.contains(childNode)) {
	console.log("Parent contains child");
}
```

### Response

#### Success Response (200)

- **boolean** - True if the node contains the child, false otherwise.

#### Response Example

```json
true
```

````

--------------------------------

### Basic sp-slider Rendering with Label

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-slider

Renders a basic slider component with an optional associated label. It requires no external dependencies and accepts 'min', 'max', and 'value' attributes for its range and current selection. The label is provided using a 'sp-label' element with the slot 'label'.

```html
<sp-slider min="0" max="100" value="50">
    <sp-label slot="label">Slider Label</sp-label>
</sp-slider>
````

---

### Synchronous vs Asynchronous Operations

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference

Explains the difference between synchronous and asynchronous operations in UXP for Premiere Pro compared to ExtendScript. It highlights that property access is synchronous while method calls are asynchronous.

```APIDOC
## Synchronous vs Asynchronous

An important difference between ExtendScript (and CEP) and UXP in Premiere is that all ExtendScript calls to Premiere were synchronous. This means they blocked the Premiere UI while they were executing.

In UXP, a method call is _asynchronous_, and does not block the UI thread.

For a smooth transition between the ExtendScript DOM and the UXP DOM, all properties (get and set) in the API were designed to be _synchronous_ and do not need to be awaited. It is worth noting that they are, in the background, asynchronous in nature.
```

---

### Prepend Nodes to Element with JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLVideoElement

Inserts a set of Node or string objects before the first child of an Element. This is a standard DOM manipulation method.

```javascript
element.prepend(...nodes);
```

---

### HTMLImageElement Properties

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLImageElement

This section outlines the readable and writable properties of the HTMLImageElement interface.

```APIDOC
## HTMLImageElement Properties

### Description
Provides access to the properties of an HTMLImageElement, such as its source, text content, and various attributes.

### Properties

- **src** (`string` | `File`) - The source of the image.
- **dataset** (`DOMStringMap`) - Access to all the custom data attributes (data-*). See: HTMLElement - dataset.
- **innerText** (`string`) - The text content of the element.
- **lang** (`string`) - Base language of an element's attribute values and text content. See: HTMLElement - lang.
- **dir** (`string`) - The text writing directionality of the content of the current element. See: HTMLElement - dir. Since: v7.1.
- **hidden** (`boolean` | `string`) - Indicates the browser should not render the contents of the element. Note: "until-found" is not supported. See: HTMLElement - hidden, Spec - `hidden` attribute.
- **nodeName** (`string`) - Read only. The name of the node.
- **localName** (`string`) - Read only. A string representing the local part of the qualified name of the element. See: https://developer.mozilla.org/en-US/docs/Web/API/Element/localName.
- **tagName** (`string`) - Read only. A string indicating the element's tag name. See: https://developer.mozilla.org/en-US/docs/Web/API/Element/tagName.
- **nodeType** (`number`) - Read only. The type of the node.
- **namespaceURI** (`string`) - Read only. Returns the namespace URI of the element, or null if the element is not in a namespace. See: https://developer.mozilla.org/en-US/docs/Web/API/Element/namespaceURI.
- **id** (`string`) - Returns the element's identifier, reflecting the id global attribute. See: https://developer.mozilla.org/en-US/docs/Web/API/Element/id.
- **tabIndex** (`number`) - The tab index of the element.
- **className** (`string`) - The class name(s) of the element.
- **attributes** (`NamedNodeMap`) - Read only. A map of the element's attributes.
- **style** (`CSSStyleDeclaration`) - Read only. The inline style of the element.
- **clientLeft** (`number`) - Read only. The distance from the element's top border edge to its top padding edge.
- **clientTop** (`number`) - Read only. The distance from the element's left border edge to its left padding edge.
- **clientWidth** (`number`) - Read only. The layout width of the element's content area, including padding but not border.
- **clientHeight** (`number`) - Read only. The layout height of the element's content area, including padding but not border.
- **offsetParent** (`Element`) - Read only. The closest ancestor element whose `position` style property is set to something other than `static`.
- **offsetLeft** (`number`) - Read only. The distance from the element's left border edge to its `offsetParent`'s left padding edge.
- **offsetTop** (`number`) - Read only. The distance from the element's top border edge to its `offsetParent`'s top padding edge.
- **offsetWidth** (`number`) - Read only. The layout width of the element, including border and padding.
- **offsetHeight** (`number`) - Read only. The layout height of the element, including border and padding.
- **scrollLeft** (`number`) - The number of pixels the element's content is scrolled horizontally.
- **scrollTop** (`number`) - The number of pixels the element's content is scrolled vertically.
- **scrollWidth** (`number`) - Read only. The scrollable width of the element's content.
- **scrollHeight** (`number`) - Read only. The scrollable height of the element's content.
- **autofocus** (`boolean`) - Indicates if the element will focus automatically when it is loaded.
- **uxpContainer** (`number`) - Read only. (Internal UXP property)
- **shadowRoot** (`ShadowRoot`) - Read only. Returns the open shadow root that is hosted by the element, or null if no open shadow root is present. See: Element - shadowRoot. [Feature flag: `enableSWCSupport`]
- **disabled** (`boolean`) - Indicates if the element is disabled.
- **innerHTML** (`string`) - Read only. Returns a fragment of the HTML, representing the element's children.
- **outerHTML** (`string`) - Returns a string containing a serialized representation of the element and its contents.
- **slot** (`string`) - The name of the slot the element belongs to. See: Element - slot. [Feature flag: `enableSWCSupport`]
- **assignedSlot** (`HTMLSlotElement`) - Read only. Returns the slot element into which the given element is inserted. See: Element - assignedSlot. [Feature flag: `enableSWCSupport`]
- **contentEditable** (`string`) - Indicates whether the element's content is editable.
- **isConnected** (`boolean`) - Read only. Indicates if the element is connected to a document.
- **parentNode** (`Node`) - Read only. The parent of the node.
- **parentElement** (`Element`) - Read only. The parent element of the node.
- **firstChild** (`Node`) - Read only. The first child of the node.
- **lastChild** (`Node`) - Read only. The last child of the node.
- **previousSibling** (`Node`) - Read only. The previous sibling of the node.
- **nextSibling** (`Node`) - Read only. The next sibling of the node.
- **firstElementChild** (`Element`) - Read only. The first element child of the node.
- **lastElementChild** (`Element`) - Read only. The last element child of the node.
- **previousElementSibling** (`Element`) - Read only. The previous element sibling of the node.
- **nextElementSibling** (`Element`) - Read only. The next element sibling of the node.
- **textContent** (`string`) - The text content of the node and its descendants.
- **childNodes** (`NodeList`) - Read only. A NodeList containing all the children of the node.
- **children** (`HTMLCollection`) - Read only. An HTMLCollection containing all the element's children.
- **ownerDocument** - Read only. The Document associated with this node.
```

---

### Icon Definition

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Defines properties for plugin icons, including dimensions, path, scaling factors, theme support, and species.

````APIDOC
## IconDefinition

### Description
Represents an icon used by the plugin or specific entry point. The icon may be used in the plugin list, toolbar, or other places.

### Properties
#### Required Properties
*   **`width`** (number) - Required - The width of the icon in pixels.
*   **`height`** (number) - Required - The height of the icon in pixels.
*   **`path`** (string) - Required - The path to the icon, relative to the plugin's installation directory. Supports PNG (`.png`), JPG (`.jpg` or `.jpeg`), and SVG (`.svg`) files.

#### Optional Properties
*   **`scale`** (number[]) - Optional - Specifies the scaling factors that the icon supports. Defaults to `[1]`.
*   **`theme`** (string[]) - Optional - Specifies the themes that the icon supports. Available themes: `"all"` (default), `"lightest"`, `"light"`, `"medium"`, `"dark"`, `"darkest"`.
*   **`species`** (string[]) - Optional - Specifies the species that the icon supports, indicating its suitable use. Available species: `"generic"`, `"toolbar"`, `"pluginList"`.

### Example (Scale)
```json
{
    "path": "icon.png",
    "width": 24,
    "height": 24,
    "scale": [1, 2, 2.5]
}
````

### Example (Theme)

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

```

```
