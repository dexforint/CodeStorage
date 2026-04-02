### Response

#### Success Response (200)

- **Action** - An object representing the action to set the in-point.

#### Response Example

```json
{
	"type": "SetInPointAction"
}
```

````

```APIDOC
## Sequence Instance Methods

### createSetOutPointAction

### Description
Creates an action to set the out-point for the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **tickTime** (_TickTime_) - The time at which to set the out-point.

### Request Example
```json
{
  "tickTime": {
    "seconds": 20,
    "ticks": 0
  }
}
````

### Response

#### Success Response (200)

- **Action** - An object representing the action to set the out-point.

#### Response Example

```json
{
	"type": "SetOutPointAction"
}
```

````

```APIDOC
## Sequence Instance Methods

### createSetSettingsAction

### Description
Returns an action that can set the sequence settings.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **sequenceSettings** (_SequenceSettings_) - The new sequence settings object.

### Request Example
```json
{
  "sequenceSettings": {
    "frameRate": 29.97,
    "videoDisplayMode": "HD1080_2997",
    "audioSampleRate": 48000
  }
}
````

### Response

#### Success Response (200)

- **Action** - An object representing the action to set sequence settings.

#### Response Example

```json
{
	"type": "SetSequenceSettingsAction"
}
```

````

```APIDOC
## Sequence Instance Methods

### createSetZeroPointAction

### Description
Creates an action to set the zero point of the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **tickTime** (_TickTime_) - The time at which to set the zero point.

### Request Example
```json
{
  "tickTime": {
    "seconds": 0,
    "ticks": 0
  }
}
````

### Response

#### Success Response (200)

- **Action** - An object representing the action to set the zero point.

#### Response Example

```json
{
	"type": "SetZeroPointAction"
}
```

````

```APIDOC
## Sequence Instance Methods

### createSubsequence

### Description
Returns a new sequence, which is a sub-sequence of the existing sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **ignoreTrackTargeting** (_boolean_) - Whether to ignore track targeting when creating the sub-sequence.

### Request Example
```json
{
  "ignoreTrackTargeting": false
}
````

### Response

#### Success Response (200)

- **Sequence** - A new sequence object representing the sub-sequence.

#### Response Example

```json
{
	"guid": "f0e9d8c7-b6a5-4321-fedc-ba9876543210",
	"name": "Sub Sequence"
}
```

````

```APIDOC
## Sequence Instance Methods

### getAudioTrack

### Description
Retrieves an audio track from the sequence by its index.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **trackIndex** (_number_) - The index of the audio track to retrieve.

### Request Example
```json
{
  "trackIndex": 0
}
````

### Response

#### Success Response (200)

- **AudioTrack** - An object representing the audio track.

#### Response Example

```json
{
	"name": "Audio 1",
	"trackType": "Audio"
}
```

````

```APIDOC
## Sequence Instance Methods

### getAudioTrackCount

### Description
Gets the total number of audio tracks in the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **number** - The count of audio tracks.

#### Response Example
```json
4
````

````

```APIDOC
## Sequence Instance Methods

### getCaptionTrack

### Description
Retrieves a caption track from the sequence by its index.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **trackIndex** (_number_) - The index of the caption track to retrieve.

### Request Example
```json
{
  "trackIndex": 0
}
````

### Response

#### Success Response (200)

- **CaptionTrack** - An object representing the caption track.

#### Response Example

```json
{
	"name": "Captions 1",
	"trackType": "Caption"
}
```

````

```APIDOC
## Sequence Instance Methods

### getCaptionTrackCount

### Description
Gets the total number of caption tracks in the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **number** - The count of caption tracks.

#### Response Example
```json
1
````

````

```APIDOC
## Sequence Instance Methods

### getEndTime

### Description
Gets the time representing the end of the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **TickTime** - The time at the end of the sequence.

#### Response Example
```json
{
  "seconds": 60,
  "ticks": 0
}
````

````

```APIDOC
## Sequence Instance Methods

### getFrameSize

### Description
Gets the size of the sequence's frame.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **RectF** - An object representing the frame size (width and height).

#### Response Example
```json
{
  "width": 1920,
  "height": 1080
}
````

````

```APIDOC
## Sequence Instance Methods

### getInPoint

### Description
Gets the time representing the in-point of the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **TickTime** - The time of the sequence's in-point.

#### Response Example
```json
{
  "seconds": 0,
  "ticks": 0
}
````

````

```APIDOC
## Sequence Instance Methods

### getOutPoint

### Description
Gets the time representing the out-point of the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **TickTime** - The time of the sequence's out-point.

#### Response Example
```json
{
  "seconds": 60,
  "ticks": 0
}
````

````

```APIDOC
## Sequence Instance Methods

### getPlayerPosition

### Description
Gets the current playback position of the sequence player.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **TickTime** - The current player position.

#### Response Example
```json
{
  "seconds": 15,
  "ticks": 0
}
````

````

```APIDOC
## Sequence Instance Methods

### getProjectItem

### Description
Gets the ProjectItem associated with the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **ProjectItem** - An object representing the ProjectItem.

#### Response Example
```json
{
  "name": "My Sequence Project Item",
  "id": "projectitem_123"
}
````

````

```APIDOC
## Sequence Instance Methods

### getSelection

### Description
Returns the current selection group of track items in the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **TrackItemSelection** - An object representing the selected track items.

#### Response Example
```json
{
  "trackItems": ["trackitem_abc", "trackitem_def"]
}
````

````

```APIDOC
## Sequence Instance Methods

### getSequenceAudioTimeDisplayFormat

### Description
Gets the audio time display format of the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **TimeDisplay** - The time display format for audio.

#### Response Example
```json
"SMPTE_2997"
````

````

```APIDOC
## Sequence Instance Methods

### getSequenceVideoTimeDisplayFormat

### Description
Gets the video time display format of the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **TimeDisplay** - The time display format for video.

#### Response Example
```json
"Frames"
````

````

```APIDOC
## Sequence Instance Methods

### getSettings

### Description
Gets the sequence settings object.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **SequenceSettings** - An object containing the sequence settings.

#### Response Example
```json
{
  "frameRate": 29.97,
  "videoDisplayMode": "HD1080_2997",
  "audioSampleRate": 48000
}
````

````

```APIDOC
## Sequence Instance Methods

### getTimebase

### Description
Gets the time base of the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **string** - The time base of the sequence (e.g., "30000/1001").

#### Response Example
```json
"30000/1001"
````

````

```APIDOC
## Sequence Instance Methods

### getVideoTrack

### Description
Retrieves a video track from the sequence by its index.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **trackIndex** (_number_) - The index of the video track to retrieve.

### Request Example
```json
{
  "trackIndex": 0
}
````

### Response

#### Success Response (200)

- **VideoTrack** - An object representing the video track.

#### Response Example

```json
{
	"name": "Video 1",
	"trackType": "Video"
}
```

````

```APIDOC
## Sequence Instance Methods

### getVideoTrackCount

### Description
Gets the total number of video tracks in the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **number** - The count of video tracks.

#### Response Example
```json
2
````

````

```APIDOC
## Sequence Instance Methods

### getZeroPoint

### Description
Gets the time representing the zero point of the sequence.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **TickTime** - The time of the sequence's zero point.

#### Response Example
```json
{
  "seconds": 0,
  "ticks": 0
}
````

````

```APIDOC
## Sequence Instance Methods

### isDoneAnalyzingForVideoEffects

### Description
Returns whether or not the sequence is done analyzing for video effects.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **boolean** - `true` if analysis is complete, `false` otherwise.

#### Response Example
```json
true
````

````

```APIDOC
## Sequence Instance Methods

### setPlayerPosition

### Description
Sets the player's current position to the specified time.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **positionTime** (_TickTime_) - The time to set the player position to.

### Request Example
```json
{
  "positionTime": {
    "seconds": 30,
    "ticks": 0
  }
}
````

### Response

#### Success Response (200)

- **boolean** - `true` if the player position was set successfully, `false` otherwise.

#### Response Example

```json
true
```

````

```APIDOC
## Sequence Instance Methods

### setSelection

### Description
Updates the sequence selection using the given track item selection.

### Method
N/A (Instance method)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **trackItemSelection** (_TrackItemSelection_) - The track item selection to apply.

### Request Example
```json
{
  "trackItemSelection": {
    "trackItems": ["trackitem_abc"]
  }
}
````

### Response

#### Success Response (200)

- **boolean** - `true` if the selection was updated successfully, `false` otherwise.

#### Response Example

```json
true
```

````

--------------------------------

### Send Binary Data via WebSocket

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/WebSocket

These JavaScript examples show how to send binary data over an established WebSocket connection using `send()`. The data can be provided as a `Float32Array` or as the buffer of an `Int32Array`. The `bufferedAmount` property tracks the amount of data queued for transmission.

```javascript
ws.send(new Float32Array([ 5, 2, 1, 3, 6, -1 ]));
````

```javascript
ws.send(new Int32Array([5, -1]).buffer);
```

---

### Run CLI Commands with UXP `openPath()`

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Use `shell.openPath()` to execute command-line interface (CLI) commands, such as shell scripts or AppleScripts. Note that this function has limitations: it cannot accept arguments for the command, and it does not capture the output. Returns an empty string on success.

```javascript
const result = await shell.openPath(
	// external command
	"/bin/ls", // ✅ will run
	"Running the ls command",
);

const result = await shell.openPath(
	// external command + argument
	"/bin/ls -la", // ❌ will fail
	"Running the ls -la command",
);
```

---

### CSS Background Property Example in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Styles/background

Demonstrates the usage of the CSS 'background' property within Adobe Premiere Pro UXP. It shows how to set a background image from plugin assets and a solid color. Note that background repeat is not supported.

```css
.someElement {
	background: url("plugin://assets/star.png") red;
}
```

---

### Check XMLHttpRequest HTTP Status Code

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

The status property returns the HTTP status code received from the server (e.g., 200 for OK, 404 for Not Found). This example logs the status code after the request completes.

```javascript
const xhr = new XMLHttpRequest();
xhr.onload = () => {
	console.log(xhr.status);
};
xhr.open("GET", "https://www.adobe.com");
xhr.send();
```

---

### Implement UXP Plugin Lifecycle Hooks and Panel Control

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-panels

This JavaScript code demonstrates how to set up UXP entrypoints, manage plugin and panel lifecycles, and programmatically show a panel. It uses lifecycle hooks to append content to the DOM when a panel is shown and an event listener to trigger the panel opening via the plugin manager.

```javascript
const { entrypoints, pluginManager } = require("uxp");

// Get reference to the second panel container
const secondPanel = document.querySelector("#second-panel");
let PLUGIN_ID;

entrypoints.setup({
	plugin: {
		create() {
			// Store the plugin ID for later use
			// (IPC -> opening the second panel programmatically)
			PLUGIN_ID = this.id;
			console.log("Plugin created:", PLUGIN_ID);
		},
	},
	panels: {
		"uxp-first-panel": {
			create() {
				console.log("First panel created");
			},
			show() {
				console.log("First panel shown");
				// The first panel content is already in the DOM
				// no other action is needed.
			},
		},
		"uxp-second-panel": {
			create() {
				console.log("Second panel created");
			},
			show(body) {
				// Append the second panel content when this panel is shown
				body.appendChild(secondPanel); // 👈 Key: append on show
				console.log("Second panel shown");
			},
			hide(body) {
				// ⚠️ Note: This hook currently doesn't work reliably
				body.removeChild(secondPanel);
				console.log("Second panel hidden");
			},
		},
	},
});

// Add button handler to open the second panel via IPC
document.querySelector("#open-second-panel").addEventListener("click", () => {
	// Find this plugin in the list of all plugins
	const me = [...pluginManager.plugins].find((plugin) => plugin.id === PLUGIN_ID);
	// Open the second panel programmatically
	me?.showPanel("uxp-second-panel"); // 👈 Opens the second panel
});
```

---

### Manipulate Element Attributes (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20DOM/Element

Provides JavaScript methods for managing HTML element attributes, including getting, setting, removing, and checking for the existence of attributes. These methods are standard DOM APIs and do not require special UXP flags.

```javascript
// Get attribute value
const attributeValue = element.getAttribute("attributeName");

// Set attribute value
element.setAttribute("attributeName", "newValue");

// Remove attribute
element.removeAttribute("attributeName");

// Check if attribute exists
const hasAttr = element.hasAttribute("attributeName");

// Get all attribute names
const attributeNames = element.getAttributeNames();

// Get attribute node
const attributeNode = element.getAttributeNode("attributeName");

// Set attribute node
element.setAttributeNode(newAttributeNode);

// Remove attribute node
element.removeAttributeNode(attributeNodeToRemove);
```

---

### Retrieve XMLHttpRequest Response as Text

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

The responseText property returns the server's response as a string. It is only available when the readyState is LOADING or DONE and the responseType is 'text' (or default). This example logs the response text upon successful completion.

```javascript
const xhr = new XMLHttpRequest();
xhr.addEventListener("load", () => {
	if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
		console.log(xhr.responseText);
	}
});
xhr.open("GET", "https://www.adobe.com");
xhr.send();
```

---

### Configure Manifest for Multiple Panels and IPC

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-panels

This JSON snippet shows how to configure the `manifest.json` file for a UXP plugin to include multiple panel entrypoints and enable inter-process communication (IPC) for programmatic control between panels. It specifies plugin details, host application compatibility, and the required `ipc` permission.

```json
{
	"id": "multi-panel-demo",
	"name": "Multi Panel Demo",
	"version": "1.0.0",
	"main": "index.html",
	"host": { "app": "premierepro", "minVersion": "25.6.0" },
	"manifestVersion": 5,
	"requiredPermissions": {
		// 👇 Required for inter-panel control
		"ipc": { "enablePluginCommunication": true }
	},
	"entrypoints": [
		{
			"id": "uxp-first-panel",
			"type": "panel",
			"label": { "default": "First Panel" },
			"minimumSize": { "width": 430, "height": 500 },
			"preferredDockedSize": { "width": 230, "height": 300 }
			// ...
		},
		{
			"id": "uxp-second-panel",
			"type": "panel",
			"label": { "default": "Second Panel" },
			"minimumSize": { "width": 430, "height": 500 },
			"preferredDockedSize": { "width": 230, "height": 300 }
			// ...
		}
	]
	// ...
}
```

---

### Element Attribute Manipulation: getAttribute, setAttribute, removeAttribute, hasAttribute, hasAttributes, getAttributeNames, getAttributeNode, setAttributeNode, removeAttributeNode

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHtmlElement

A comprehensive set of methods for managing element attributes. This includes getting, setting, removing, and checking for the existence of attributes by name or node.

```javascript
/**
 * Gets the value of a specified attribute on the element.
 * @param {string} name - Name of the attribute whose value you want to get.
 * @returns {string} The value of the attribute.
 */
function getAttribute(name) {
	// Implementation details for getting an attribute
}

/**
 * Sets the value of a specified attribute on the element.
 * @param {string} name - Name of the attribute whose value is to be set.
 * @param {string} value - Value to assign to the attribute.
 */
function setAttribute(name, value) {
	// Implementation details for setting an attribute
}

/**
 * Removes a specified attribute from the element.
 * @param {string} name - Name of the attribute to remove.
 */
function removeAttribute(name) {
	// Implementation details for removing an attribute
}

/**
 * Checks if the element has a specified attribute.
 * @param {string} name - Name of the attribute to check for.
 * @returns {boolean} True if the attribute exists, false otherwise.
 */
function hasAttribute(name) {
	// Implementation details for checking attribute existence
}

/**
 * Returns a boolean value indicating whether the current element has any attributes or not.
 * @returns {boolean} True if the element has any attributes, false otherwise.
 */
function hasAttributes() {
	// Implementation details for checking if any attributes exist
}

/**
 * Returns the attribute names of the element as an Array of strings.
 * @returns {Array<string>} An array containing the names of all attributes.
 */
function getAttributeNames() {
	// Implementation details for getting all attribute names
}

/**
 * Gets the attribute node with the specified name.
 * @param {string} name - The name of the attribute node to retrieve.
 * @returns {*} The attribute node.
 */
function getAttributeNode(name) {
	// Implementation details for getting an attribute node
}

/**
 * Sets the attribute node for the element.
 * @param {*} newAttr - The attribute node to set.
 */
function setAttributeNode(newAttr) {
	// Implementation details for setting an attribute node
}

/**
 * Removes the specified attribute node from the element.
 * @param {*} oldAttr - The attribute node to remove.
 */
function removeAttributeNode(oldAttr) {
	// Implementation details for removing an attribute node
}
```

---

### Handle Radio Button Click Event (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-radio

Provides a JavaScript example for handling the 'click' event on a radio button. It logs the value of the clicked radio button to the console, allowing for dynamic interaction with user selections.

```javascript
document.querySelector(".yourRadioButton").addEventListener("click", (evt) => {
	console.log(`You clicked: ${evt.target.value}`);
});
```

---

### Get Temporary Folder (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider

Shows how to obtain a temporary folder using `getTemporaryFolder()`. Contents of this folder are automatically removed when the UXP extension is disposed, making it suitable for transient data.

```javascript
const temp = await fs.getTemporaryFolder();
```

---

### Accessing Globally Available UXP Core APIs (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis

Demonstrates how to access globally available UXP Core APIs like the Crypto API for generating UUIDs and the Document API for creating HTML elements. These APIs do not require explicit imports.

```javascript
// Crypto API is globally available
const hash = crypto.randomUUID();

// Document API is globally available
const button = document.createElement("sp-button");
```

---

### Create ImageBlob using PhotoshopImageData

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/ImageBlob

This example shows how to create an ImageBlob by utilizing a `PhotoshopImageData` object, which is compatible with the `ImageBlob` constructor. It assumes that `photoshopImageObject` is already obtained through Photoshop's imaging APIs. The `getData()` method is called on this object to retrieve the image data as an `ArrayBuffer`, which is then used to instantiate `ImageBlob`. This method is asynchronous and requires the `await` keyword.

```javascript
// Creating ImageBlob using PhotoshopImageData object(more details about PhotoshopImageData in description).
const photoshopImageObject; // get image object using Photoshp's imaging apis.
let colorArrayView = await photoshopImageObject.getData();
```

---

### Invalid WebView Domain Configuration in UXP Manifest

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

This example illustrates an invalid configuration for the `domains` property within the WebView permissions in a UXP plugin's `manifest.json`. It highlights that top-level wildcards are not permitted for domain matching, which can lead to the plugin being blocked.

```json
"requiredPermissions": {
 "webview": {
     "domains": ["https://www.*", "https://www.adobe.*"],
     "allow": "yes"
  }
}
```

---

### Getting Native File System Path (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/Entry

Illustrates how to retrieve the platform-specific native file system path of an entry using the read-only `nativePath` string property.

```javascript
console.log(anEntry.nativePath);
```

---

### Read Text from Clipboard (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/clipboard

This example shows how to read plain text content from the system clipboard using `navigator.clipboard.getContent()`. It requires the `read` clipboard permission in `manifest.json`. The function attempts to retrieve text data and logs it or indicates if no text is found. Error handling is included.

```javascript
// Paste text from the clipboard 📋
async function pasteFromClipboard() {
	try {
		const clipboardData = await navigator.clipboard.getContent();

		if (clipboardData["text/plain"]) {
			console.log(`Pasted text: ${clipboardData["text/plain"]} `);
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

```json
{
	// ...
	"requiredPermissions": {
		"clipboard": "read"
	}
	// ...
}
```

---

### open(method, url, [async], [user], [password])

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

Initializes a request or re-initializes an existing one. Self-signed certificates are not currently supported for HTTPS connections. UXP does not support synchronous requests.

```APIDOC
## open(method, url, [async], [user], [password])

### Description
Initializes a request or re-initializes an existing one. Self-signed certificates are not currently supported for HTTPS connections. Note that UXP does not support synchronous request, which means 'async' is false.

**Throws** :
  * `DOMException` NotSupportedError if async parameter is false
  * `DOMException` TypeError if method and url parameters are not provided
  * `DOMException` SyntaxError if either method is not valid or url cannot be parsed.
  * `DOMException` SecurityError if method matches for CONNECT, TRACE or TRACK.

### Parameters
#### Path Parameters
- **method** (string) - Required - HTTP request method to use, such as "GET", "POST", "PUT", "DELETE", etc. Ignored for non-HTTP(S) URLs.
- **url** (string) - Required - String representing the URL to send the request to.
- **async** (boolean) - Optional - Defaults to `true`. Indicates whether or not to perform the operation asynchronously. If this value is false, the send() method does not return until the response is received. If true, notification of a completed transaction is provided using event listeners. This must be true if the multipart attribute is true, or an exception will be thrown.
- **user** (string) - Optional - Defaults to `null`. User name to use for authentication purposes.
- **password** (string) - Optional - Defaults to `null`. Password to use for authentication purposes.
```

---

### Rename an Entry in a Folder (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/Folder

Provides an example of how to rename an existing file or folder entry within its parent folder. It includes an option to overwrite if an entry with the new name already exists.

```javascript
await myNovels.rename(myNovel, "myFantasticNovel.txt");
```

---

### Set Request Header for XMLHttpRequest

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

Adds or appends a value to a request header for an XMLHttpRequest. Forbidden headers, including those starting with 'proxy-' or 'sec-', are ignored. This method requires the request state to be OPENED and send() not to have been invoked.

```javascript
const xhr = new XMLHttpRequest();
xhr.open("GET", "https://www.mywebserver.com");
xhr.setRequestHeader("Accept", "text/xml");
xhr.send();
```

---

### Apply Top Border Color CSS

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Styles/border-top-color

This CSS snippet demonstrates how to set the top border color for an element using the 'border-top-color' property. It requires the 'border-style' property to be defined as well. The example uses a named color 'blue'.

```css
.someElement {
	border-style: solid;
	border-top-color: blue;
}
```

---

### Associate Command Entrypoint with Handler using module.exports

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-commands

This method uses `module.exports` in `index.js` to associate command entrypoints with their handlers. It's a concise option for plugins that exclusively use command entrypoints and do not require a UI file. The `main` property in `manifest.json` must point to this `index.js` file.

```javascript
module.exports = {
	commands: {
		myCommand: myCommandHandler,
	},
};
```

---

### Get Data Folder (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider

Provides access to a persistent data folder for the extension using `getDataFolder()`. This folder is suitable for storing extension data that should persist across application upgrades and does not require user interaction.

```javascript
const dataFolder = await fs.getDataFolder();
```

---

### WebView Load Error Event Listener (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

Provides an example of capturing the 'loaderror' event from a WebView. This event is triggered when a page fails to load, and the event object contains the URL, an error code, and a descriptive message.

```javascript
// Print the url, code and message when loading has failed
webview.addEventListener("loaderror", (e) => {
	console.log(`webview.loaderror ${e.url}, code:${e.code}, message:${e.message}`);
});
```

---

### Get Specific Response Header with XMLHttpRequest

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

Retrieves the value of a specific response header, specified by its name. The search is case-insensitive. This method throws a TypeError if the name parameter is not provided. It's often used when readyState is XMLHttpRequest.HEADERS_RECEIVED.

```javascript
const xhr = new XMLHttpRequest();
xhr.onreadystatechange = () => {
	if (xhr.readyState === XMLHttpRequest.HEADERS_RECEIVED) {
		console.log(xhr.getResponseHeader("Content-Type"));
	}
};
xhr.open("GET", "https://www.adobe.com");
xhr.send();
```

---

### Listen for UXP Command Events

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/entrypoints

Demonstrates how to attach an event listener to the `document.body` to capture `uxpcommand` events. This is an alternative method for handling command invocations within a UXP plugin.

```javascript
document.body.addEventListener("uxpcommand" (event) => { /* ... */ });

```

---

### Create Directory Asynchronously (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/fs

Creates a directory at the specified path asynchronously. Returns a Promise that resolves with 0 on success or throws an error. Accepts a path string and optional options, including a recursive flag to create parent directories.

```javascript
await fs.mkdir("plugin-data:/newDir");
```

---

### Clipboard API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/Clipboard

The Clipboard API provides methods to interact with the system's clipboard. This includes functionalities to set, get, write, and read data from the clipboard, as well as clear its content. Note that some methods are non-standard or have specific version requirements.

````APIDOC
## Clipboard() Constructor

### Description
Creates an instance of Clipboard. Note: Clipboard access is not supported for 3P plugins with manifest version up to 4. A valid manifest entry is required from manifest version 5.

### Method
Constructor

### Endpoint
N/A (JavaScript API)

### Parameters
None

### Request Example
```javascript
const clipboard = new Clipboard();
````

### Response

#### Success Response

An instance of the Clipboard object.

#### Response Example

N/A

````

```APIDOC
## setContent(data)

### Description
Set data to the clipboard. This is a non-standard API.

### Method
`navigator.clipboard.setContent(data)`

### Endpoint
N/A (JavaScript API)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **data** (object) - Required - The data to store in the clipboard. The object keys are the mime types, so for text, use `text/plain` as a key.

### Request Example
```javascript
navigator.clipboard.setContent({"text/plain": "Hello!"});
````

### Response

#### Success Response (Promise)

Resolves when the data is set to the clipboard.

#### Response Example

```json
// Promise resolves with no specific value on success
```

````

```APIDOC
## getContent()

### Description
Get data from the clipboard. This is a non-standard API.

### Method
`navigator.clipboard.getContent()`

### Endpoint
N/A (JavaScript API)

### Parameters
None

### Request Example
```javascript
navigator.clipboard.getContent();
````

### Response

#### Success Response (Promise)

Resolves with the data from the clipboard.

#### Response Example

```json
{
	"text/plain": "Pasted content"
}
```

````

```APIDOC
## write(data)

### Description
Write data to the clipboard. This can be used to implement cut and copy functionality. Available since v6.0.

### Method
`navigator.clipboard.write(data)`

### Endpoint
N/A (JavaScript API)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **data** (object) - Required - The data to set. Example: `{"text/plain": "Hello!"}`

### Request Example
```javascript
navigator.clipboard.write({"text/plain": "Hello!"});
````

### Response

#### Success Response (Promise)

Resolves when the data is written to the clipboard.

#### Response Example

```json
// Promise resolves with no specific value on success
```

````

```APIDOC
## writeText(text)

### Description
Write text to the clipboard. This can be used to implement cut and copy text functionality. Available since v6.0.

### Method
`navigator.clipboard.writeText(text)`

### Endpoint
N/A (JavaScript API)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **text** (string | object) - Required - The text string to set or an object of the form `{"text/plain": "text to set"}`. Note that the object format will be deprecated and shouldn't be used.

### Request Example
```javascript
navigator.clipboard.writeText("Hello!");
````

### Response

#### Success Response (Promise)

Resolves when the text is written to the clipboard.

#### Response Example

```json
// Promise resolves with no specific value on success
```

````

```APIDOC
## read()

### Description
Read data from the clipboard. Available since v6.0.

### Method
`navigator.clipboard.read()`

### Endpoint
N/A (JavaScript API)

### Parameters
None

### Request Example
```javascript
navigator.clipboard.read();
````

### Response

#### Success Response (Promise)

Resolves with the data read from the clipboard.

#### Response Example

```json
{
	"text/plain": "Pasted content"
}
```

````

```APIDOC
## readText()

### Description
Read text from the clipboard. Available since v6.0.

### Method
`navigator.clipboard.readText()`

### Endpoint
N/A (JavaScript API)

### Parameters
None

### Request Example
```javascript
navigator.clipboard.readText();
````

### Response

#### Success Response (Promise)

Resolves with the text read from the clipboard.

#### Response Example

```json
"Pasted text"
```

````

```APIDOC
## clearContent()

### Description
Clear clipboard content. Note: This method is non-standard. Available since v6.0.

### Method
`navigator.clipboard.clearContent()`

### Endpoint
N/A (JavaScript API)

### Parameters
None

### Request Example
```javascript
navigator.clipboard.clearContent();
````

### Response

#### Success Response (Promise)

Resolves when the clipboard content is cleared.

#### Response Example

```json
// Promise resolves with no specific value on success
```

````

--------------------------------

### Getting Entry URL (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/Entry

Shows how to access the read-only `url` string property of an entry. This URL can be used with other UXP components, such as setting the `src` attribute of an Image widget.

```javascript
console.log(anEntry.url);
````

---

### Retrieve XMLHttpRequest Response as XML/HTML Document

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

The responseXML property returns the server's response as a Document object, suitable for parsing XML or HTML. This requires setting the responseType to 'document'. The example logs the response document if the request is successful.

```javascript
const xhr = new XMLHttpRequest();
xhr.onload = () => {
	if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
		console.log(xhr.responseXML);
	}
};
xhr.open("GET", "https://www.mydocumentserver.com");
xhr.responseType = "document";
xhr.send();
```

---

### Add JSDoc Type Hints for Premiere Pro DOM APIs in JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/typescript-support

Demonstrates how to add JSDoc type hints to JavaScript code for Premiere Pro UXP plugins. It shows how to type the main `ppro` object for automatic type flow and provides examples for function parameters, return types, and complex types like arrays.

```javascript
/** @type {import('./types').premierepro} */
const ppro = require("premierepro");

// Now you get full IntelliSense!
ppro.Project.getActiveProject(); // ✅ Autocomplete works!
```

```javascript
/** @type {import('./types').premierepro} */
const ppro = require("premierepro");

async function example() {
	const project = await ppro.Project.getActiveProject();
	const sequence = await project.getActiveSequence();

	// Types flow through automatically; sequence has full IntelliSense!
	const trackCount = await sequence.getVideoTrackCount();
}
```

```javascript
/**
 * Analyze all video tracks in a sequence
 * @param {import('./types').Sequence} sequence - The sequence to analyze
 * @returns {Promise<void>}
 */
async function analyzeTracks(sequence) {
	const videoTrackCount = await sequence.getVideoTrackCount();

	for (let i = 0; i < videoTrackCount; i++) {
		const track = await sequence.getVideoTrack(i);
		console.log(`Track ${i}: ${track.name}`);
	}
}
```

```javascript
/**
 * Process multiple track items
 * @param {import('./types').VideoClipTrackItem[]} trackItems
 */
async function processTrackItems(trackItems) {
	for (const item of trackItems) {
		// item has full IntelliSense
		const name = await item.getName();
		console.log(name);
	}
}
```

---

### Basic HTML Body Structure for UXP Plugin Panel

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-html/General/body

This snippet demonstrates the fundamental HTML structure for the main body of a UXP plugin panel. It shows a simple 'Hello, world' message within the body tag, serving as the initial content for the first panel.

```html
<!DOCTYPE html>
<html>
	<head>
		<title>My UXP Plugin</title>
	</head>
	<body>
		Hello, world
	</body>
</html>
```

---

### Get Active Sequence from Project in Premiere Pro

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference

Obtains the currently active sequence from a given project object in Adobe Premiere Pro. This asynchronous operation returns a sequence object, allowing further manipulation of timeline data.

```javascript
const sequence = await project.getActiveSequence();
```

---

### Get File for Saving (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider

Illustrates how to use `getFileForSaving` to obtain a file reference for read-write operations, allowing the user to choose a save location. It includes specifying a suggested name and allowed file types.

```javascript
const file = await fs.getFileForSaving("output.txt", { types: ["txt"] });
if (!file) {
	// file picker was cancelled
	return;
}
await file.write("It was a dark and stormy night");
```

---

### Handling sp-slider Input Events in JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-slider

Provides a JavaScript example for responding to changes in the sp-slider's value. It attaches an event listener to the 'input' event on a slider identified by a class '.yourSlider'. The new value is logged to the console.

```javascript
document.querySelector(".yourSlider").addEventListener("input", (evt) => {
	console.log(`New value: ${evt.target.value}`);
});
```

---

### Get Active Project in Premiere Pro

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference

Retrieves the currently active project within Adobe Premiere Pro using the UXP API. This function is asynchronous and returns a project object that can be used to access sequences and other project-specific data.

```javascript
const project = await app.Project.getActiveProject();
```

---

### Create and Serialize New XMP Metadata

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/getting-started

This code creates a new, empty XMPMeta object, sets a 'CreatorTool' property within the XMPConst.NS_XMP namespace, and then serializes the metadata packet into an XML string. It also shows how to retrieve and log the created property.

```javascript
const { XMPMeta, XMPConst } = require("uxp").xmp;
let xmp = new XMPMeta();
xmp.setProperty(XMPConst.NS_XMP, "CreatorTool", "My Script");
let xmpStr = xmp.serialize(); // Serialize the XMP packet to XML

// Retrieve property
let prop = xmp.getProperty(XMPConst.NS_XMP, "CreatorTool");
console.log(`namespace: ${prop.namespace}, property path + name: ${prop.path}, value: ${prop.value}`);
```

---

### Apply Left Padding with CSS in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Styles/padding-left

Demonstrates how to apply left padding to an element using the 'padding-left' CSS property within Adobe Premiere Pro UXP. This property is available since UXP v2.0. The example shows a basic CSS rule.

```css
.someElement {
	padding-left: 10px;
}
```

---

### Specify Host Application Compatibility in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Defines the host applications and their version ranges that a UXP plugin supports. The 'app' property specifies the host (e.g., 'PS', 'premierepro'), while 'minVersion' and 'maxVersion' define the supported version range.

```json
{
	"app": "premierepro",
	"minVersion": "15.0.0",
	"maxVersion": "16.5.0"
}
```

---

### Accessing Premiere APIs with require() (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis

Illustrates how to access Premiere APIs by requiring the 'premierepro' module. The 'app' object, obtained through this require statement, serves as the entry point for interacting with Premiere Pro's document model.

```javascript
const app = require("premierepro");
```

---

### Keyframe Instance Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/keyframe

Details the instance methods available for Keyframe objects.

```APIDOC
## Keyframe Instance Methods

### getTemporalInterpolationMode

#### Description
Gets temporal interpolation mode of a keyframe.

#### Method
GET

#### Endpoint
`/keyframe/getTemporalInterpolationMode`

#### Returns
- **temporalInterpolationMode** (number) - The temporal interpolation mode.

### setTemporalInterpolationMode

#### Description
Sets temporal interpolation mode of a keyframe.

#### Method
POST

#### Endpoint
`/keyframe/setTemporalInterpolationMode`

#### Parameters
##### Request Body
- **temporalInterpolationMode** (number) - Required - The temporal interpolation mode to set.
```

---

### Disabled sp-slider State

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-slider

Demonstrates how to render a disabled sp-slider. This state prevents user interaction and is indicated by the 'disabled' attribute. It retains the 'min', 'max', and 'value' attributes for initial setup.

```html
<sp-slider
	disabled
	min="0"
	max="100"
	value="50"
></sp-slider>
```

---

### Dynamically Load Second Panel HTML with JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-panels

Demonstrates how to fetch the HTML content of a second panel from a separate file (`second-panel.html`) and inject it into a designated DOM element (`#second-panel`) using JavaScript's `fetch` API. This approach is suitable for organizing complex UXP plugins.

```javascript
const { entrypoints, pluginManager } = require("uxp");

// Get reference to the second panel container
const secondPanel = document.querySelector("#second-panel");

// Fetch the second panel's HTML file and inject the content into the DOM
fetch("./second-panel.html") // ←
	// Convert the response to text                         // ←
	.then((response) => response.text()) // ←
	// Set the content of the second panel container        // ←
	.then((html) => {
		// ←
		secondPanel.innerHTML = html; // ←
		// Add event listeners for the second panel here      // ←
		// or any other JavaScript code you need              // ←
		// ...                                                // ←
	}); // ←

let PLUGIN_ID;
entrypoints.setup({
	/* ... */
});

// Everything else is the same as in the previous example...
```

---

### ProjectSettings - createSetIngestSettingsAction

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/projectsettings

Returns an action which sets IngestSettings for a project.

```APIDOC
## POST /websites/developer_adobe_premiere-pro_uxp/ProjectSettings/createSetIngestSettingsAction

### Description
Returns an action which sets IngestSettings for a project.

### Method
POST

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/ProjectSettings/createSetIngestSettingsAction

### Parameters
#### Request Body
- **project** (_Project_) - Required - The project for which to set ingest settings.
- **ingestSettings** (_IngestSettings_) - Required - The ingest settings to apply.

### Request Example
{
  "project": "project_object",
  "ingestSettings": "ingest_settings_object"
}

### Response
#### Success Response (200)
- **action** (_Action_) - An action object that can be invoked to set the ingest settings.

#### Response Example
{
  "action": "action_object"
}
```

---

### UXP Command Entry Point Functions in JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Entry%20Points/EntryPoints

This documentation describes the entry point functions for UXP commands. `entrypoints.commands.run` is executed when a command is invoked, receiving an event object. It supports asynchronous operations via promises and error handling through exceptions or rejected promises. `entrypoints.commands.cancel` is reserved for future use.

```javascript
// Called when the command is invoked via menu entry.
// 'this' can be used to access UxpCommandInfo object.
// This function can return a promise.
// To signal failure, throw an exception or return a rejected promise.
// Parameters :
// run(event) {} // till Manifest Version V4
// run(executionContext, ...arguments) {} // from v5 onwards

// For future use.
// entrypoints.commands.cancel
```

---

### JavaScript for Premiere Pro UXP Modal Dialog (Singleton Pattern)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-modal-dialogs

Implements the Singleton pattern for a modal dialog in UXP. It manages dialog creation, content loading from fragments, event listeners, and input validation using JavaScript. Dependencies include the UXP API and fetched HTML/CSS content.

```javascript
// Global constants for the dialog
const G = {
  version: "1.0.0",
  title: "Input required...",
  dialogSize: { width: 240, height: 150 },
  defaultValues: { width: 1920, height: 1080 },
  valueRange: { min: 320, max: 10240 },
};


class ModalDialog {
  static #instance; // Singleton instance
  #dialog;          // Reference to the dialog element
  // State container for validated values
  #params = { width: 0, height: 0 };


  constructor() {
    // Enforce singleton pattern
    if (ModalDialog.#instance) { return ModalDialog.#instance; }
    ModalDialog.#instance = this;
  }


  // Static getter for singleton instance (optional, cleaner API)
  static getInstance() {
    if (!ModalDialog.#instance) {
      ModalDialog.#instance = new ModalDialog();
    }
    return ModalDialog.#instance;
  }


  // Create and populate a dialog w/ HTML elements
  // Assign to #dialog
  // Uses the same pattern as styles: create once, reuse thereafter
  async createDialog() {
    // Add scoped styles to the head
    // The conditional check ensures the styles are added only once
    // to avoid duplicates when the dialog is opened multiple times
    if (!document.querySelector("#modal-dialog-styles")) {
      const styleEl = document.createElement("style");
      styleEl.id = "modal-dialog-styles";
      styleEl.textContent = (
        await fetch("./fragments/styles.css").then((res) => res.text())
      ).trim();
      document.head.appendChild(styleEl);
    }


    // Same pattern for the dialog: create once, reuse thereafter
    if (!document.querySelector("#modal-dialog")) {
      this.#dialog = document.createElement("dialog");
      this.#dialog.id = "modal-dialog";
      // Add unique class for scoping the CSS
      this.#dialog.classList.add("modal-dialog");
      this.#dialog.innerHTML = (
        await fetch("./fragments/dialog.html").then((res) => res.text())
      ).trim();
      document.body.appendChild(this.#dialog);
    } else {
      // Reuse existing dialog
      this.#dialog = document.querySelector("#modal-dialog");
    }
  }


  // Populate w/ default values
  // Attach event listeners (only once)
  initDialog() {
    const wField = this.#dialog.querySelector("#width");
    const hField = this.#dialog.querySelector("#height");


    // Always reset to default values on each init
    wField.value = G.defaultValues.width.toString();
    hField.value = G.defaultValues.height.toString();


    // Attach event listeners only once; check the dialog element itself
    // since the dialog persists but new instances are created each time
    if (!this.#dialog.dataset.listenersAttached) {
      // Sanitize input for the textfields
      const sanitizeInput = (evt) => {
        /* ... */
      };
      wField.addEventListener("input", sanitizeInput);
      hField.addEventListener("input", sanitizeInput);


      // Validate and return params object if valid, null if invalid
      const validateAndGetParams = () => {
        const w = parseInt(wField.value);
        const h = parseInt(hField.value);
        if (
          isNaN(w) || isNaN(h) ||
          w < G.valueRange.min || w > G.valueRange.max ||
          h < G.valueRange.min || h > G.valueRange.max
        ) {
          return null;
        }
        return { width: w, height: h };
      };


      this.#dialog

```

---

### Get File/Folder Stats Synchronously (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/fs

Retrieves file or folder information synchronously using the `lstatSync` method. It returns a stats object similar to Node.js's stats class, though some properties might be platform-limited. Requires a string path as input.

```javascript
const stats = fs.lstatSync("plugin-data:/textFile.txt");
const birthTime = stats.birthtime;
```

---

### Manifest Configuration for About Command (Premiere Pro UXP)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-modal-dialogs

This JSON configuration defines the 'About...' command for a UXP plugin in Adobe Premiere Pro. It specifies the command's ID, type, and user-facing label, linking it to the JavaScript implementation defined in main.js.

```json
{
	"id": "Test-modaldialog",
	"name": "Test-modaldialog",
	"version": "1.0.0",
	"main": "main.js",
	"host": { "app": "premierepro", "minVersion": "25.6.0" },
	"manifestVersion": 5,
	"entrypoints": [
		{
			"id": "about-command",
			"type": "command",
			"label": "About..."
		}
	],
	"icons": [
		/* ... icons ... */
	]
}
```

---

### Encode URL Parameters (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Demonstrates the use of `encodeURIComponent()` to properly encode special characters within URL parameters, specifically for the subject of a `mailto` link. This prevents issues with URL parsing and ensures the link functions as intended.

```javascript
const subject = encodeURIComponent("My Subject");
const url = `mailto:user@example.com?subject=${subject}`;
```

---

### WritableStreamDefaultWriter API Documentation

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Streams/WritableStreamDefaultWriter

This section covers the WritableStreamDefaultWriter constructor, properties like 'closed', 'desiredSize', and 'ready', and methods such as 'abort', 'close', 'releaseLock', and 'write'.

````APIDOC
## WritableStreamDefaultWriter(stream)

### Description
Creates a new WritableStreamDefaultWriter object.

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
None

### Request Example
```javascript
const writer = new WritableStreamDefaultWriter(stream);
````

### Response

#### Success Response (200)

Returns a `WritableStreamDefaultWriter` object.

#### Response Example

```json
{
	"type": "WritableStreamDefaultWriter"
}
```

## closed : `Promise<void>`

### Description

Returns a Promise that fullfils if the stream becomes closed, or rejects if the stream errors or the writer's lock is released.

### Method

Getter

### Endpoint

N/A

### Parameters

None

### Request Example

```javascript
writer.closed.then(() => console.log("Stream is closed"));
```

### Response

#### Success Response (200)

A `Promise<void>` that resolves when the stream is closed.

#### Response Example

```json
null
```

## desiredSize : `number`

### Description

The desired size required to fill the stream's internal queue. It will return null if the stream cannot be successfully written to. It will return zero if the stream is closed.

### Method

Getter

### Endpoint

N/A

### Parameters

None

### Request Example

```javascript
const size = writer.desiredSize;
```

### Response

#### Success Response (200)

A `number` representing the desired size, or `null`.

#### Response Example

```json
1024
```

## ready : `Promise<void>`

### Description

Returns a Promise that resolves when the desired size of the stream's internal queue transitions from non-positive to positive, signaling that it is no longer applying backpressure. Once the desired size dips back to zero or below, the getter will return a new promise that stays pending until the next transition. If the stream becomes errored or aborted, or the writer’s lock is released, the returned promise will become rejected.

### Method

Getter

### Endpoint

N/A

### Parameters

None

### Request Example

```javascript
writer.ready.then(() => console.log("Stream is ready for writing"));
```

### Response

#### Success Response (200)

A `Promise<void>` that resolves when the stream is ready.

#### Response Example

```json
null
```

## abort(reason)

### Description

Aborts the stream, signaling that the producer can no longer successfully write to the stream and it is to be immediately moved to an errored state, with any queued-up writes discarded. The returned promise will fulfill if the stream shuts down successfully, or reject if the underlying sink signaled that there was an error doing so. It will reject with a TypeError if the stream is curretly locked.

### Method

`abort`

### Endpoint

N/A

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

- **reason** (string) - Required - The reason for aborting the stream.

### Request Example

```javascript
writer.abort("User cancelled operation");
```

### Response

#### Success Response (200)

A `Promise<void>` that fulfills if the stream shuts down successfully.

#### Response Example

```json
null
```

## close()

### Description

Closes the stream and returns a Promise that will fulfill if all remaining chunks are successfully written and the stream successfully closes, or rejects if an error is encountered during this process. It will reject with a TypeError (without attempting to cancel the stream) if the stream is currently locked.

### Method

`close`

### Endpoint

N/A

### Parameters

None

### Request Example

```javascript
writer.close();
```

### Response

#### Success Response (200)

A `Promise<void>` that fulfills if the stream closes successfully.

#### Response Example

```json
null
```

## releaseLock()

### Description

Releases the writer’s lock on the corresponding stream. After the lock is released, the writer is no longer active. If the associated stream is errored when the lock is released, the writer will appear errored in the same way from now on; otherwise, the writer will appear closed.

### Method

`releaseLock`

### Endpoint

N/A

### Parameters

None

### Request Example

```javascript
writer.releaseLock();
```

### Response

#### Success Response (200)

No explicit return value, but the writer's lock is released.

#### Response Example

```json
null
```

## write(chunk)

### Description

Writes the given chunk to the writable stream and its underlying sink, then returns a Promise that resolves to indicate the success or failure of the write operation.

### Method

`write`

### Endpoint

N/A

### Parameters

#### Path Parameters

None

#### Query Parameters

None

#### Request Body

- **chunk** (\*) - Required - The data chunk to write to the stream.

### Request Example

```javascript
writer.write("Hello, world!");
```

### Response

#### Success Response (200)

A `Promise<void>` that resolves upon successful write.

#### Response Example

```json
null
```

````

--------------------------------

### Access XMLHttpRequest Response Body

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

The response property provides access to the response's body content in its native format (string, ArrayBuffer, Blob, Document, or Object). It is null if the request is incomplete or failed. This example logs the response body after a successful request.

```javascript
const xhr = new XMLHttpRequest();
xhr.onload = () => {
    if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
        console.log(xhr.response);
    }
};
xhr.open("GET", "https://www.adobe.com");
xhr.send();
````

---

### Instantiate WebSocket Connection

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/WebSocket

This JavaScript code demonstrates how to create a new WebSocket connection to a server. It takes the server URL and optionally a list of protocols as arguments. An error is thrown if the URL or protocols are invalid.

```javascript
var ws = new WebSocket("wss://demos.kaazing.com/echo", "wss");
```

---

### dumpNamespaces - Dump Registered Namespaces

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPMeta

Creates and returns a human-readable string listing all registered namespace URIs and their associated prefixes.

````APIDOC
## dumpNamespaces()

### Description
Creates and returns a human-readable string containing the list of registered namespace URIs and their associated prefixes.

### Method
`dumpNamespaces`

### Parameters
None

### Request Example
```javascript
XMPMeta.dumpNamespaces()
````

### Response

#### Success Response (200)

- **Return Value** (string) - A string listing the registered namespace URIs and their associated prefixes.

#### Response Example

```json
{
	"return": "<ns1>: http://example.com/ns1\n<ns2>: http://example.com/ns2"
}
```

````

--------------------------------

### Apply Styles When Element is Active in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Pseudo-classes/active

The :active pseudo-class allows you to style elements while they are being clicked by the user. This is useful for providing visual feedback during user interaction. The example demonstrates changing the background color of a paragraph element when it is active.

```css
p:active {
    background-color: yellow;
}
````

---

### Apply styles when element is focused using :focus CSS

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Pseudo-classes/focus

The :focus CSS pseudo-class selects an element when it has received focus. For non-interactive elements, a positive tab index is required. This example demonstrates applying a red border to a focused input element.

```css
input:focus {
	border: 1px solid red;
}
```

---

### Configure main property for module.exports in manifest.json

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-commands

When using the `module.exports` method for command handlers, the `main` property in `manifest.json` must be set to the path of the `index.js` file containing the exports. This tells UXP where to find the plugin's main entry point.

```json
{
	// ...
	"main": "index.js",
	// ...
	"entrypoints": [
		{
			"type": "command",
			"id": "myCommand",
			"label": "This is a Command"
		}
	]
	// ...
}
```

---

### Render Horizontal Rule with hr Element (HTML)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-html/Hierarchy/hr

Demonstrates how to render a horizontal rule using the 'hr' HTML element in UXP. It shows examples with 'large', 'medium', and 'small' classes for different visual sizes. The 'hr' element is not theme-aware.

```HTML
<hr class="large" />
<hr class="medium" />
<hr class="small" />
```

---

### Define Entrypoint Shortcut in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Specifies keyboard shortcuts for UXP entrypoints, with platform-specific definitions for macOS and Windows. Shortcuts must adhere to a defined syntax including modifier keys and a single letter or number key. Collisions with existing shortcuts will result in warnings.

```javascript
{
    "shortcut": {
        "mac": "Cmd+Shift+P",
        "win": "Ctrl+Shift+P"
    }
}
```

---

### Creating a Text Layer using Premiere APIs (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis

Shows the basic steps to create a text layer within the active sequence of a Premiere Pro project using Premiere APIs. It accesses the active sequence via the 'app' object.

```javascript
const app = require("premierepro");
const sequence = app.project.activeSequence;
// Add text layer to sequence
```

---

### Console Logging in UXP JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/debug

Demonstrates how to use `console.log`, `console.warn`, and `console.error` to output information, warnings, and errors to the UXP Developer Tool console. It also shows how to log variables, objects, and formatted strings using template literals for effective debugging.

```javascript
// Basic logging
console.log("Plugin initialized"); // 💡 General information
console.warn("This feature is deprecated"); // ⚠️ Warnings (yellow)
console.error("Failed to load data"); // ❌ Errors (red)

// Log variables and objects
const user = { name: "Jane", role: "Editor" };
console.log("User data:", user);
// Logs: User data: { name: "Jane", role: "Editor" }

// Log multiple values using template literals
const width = 1920;
const height = 1080;
console.log(`Resolution: ${width}x${height}`); // Template literals for formatting
```

---

### Icon Configuration with Scaling (JSON)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Defines an icon for a plugin entry point, specifying its path, dimensions, and supported scaling factors. The system selects the appropriate icon file based on the device's scale.

```json
{
	"path": "icon.png",
	"width": 24,
	"height": 24,
	"scale": [1, 2, 2.5]
}
```

---

### HTML Structure for Multiple Panels

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-panels

Defines the HTML structure for a UXP plugin with multiple panels. The main `index.html` includes a wrapper for the first panel and an empty placeholder div with an ID for the second panel, which will be dynamically loaded.

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

		<!-- Second panel wrapper -->
		<div
			class="wrapper"
			id="second-panel"
		>
			<!-- 👁️ 👁️ nothing here yet 👁️ 👁️ -->
		</div>
	</body>
</html>
```

---

### Set background-image CSS - UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Styles/background-image

Demonstrates how to set a background image using the CSS 'background-image' property in Adobe Premiere Pro UXP. This example specifies a local asset and highlights the UXP asset path syntax. Note that background repeat is not supported.

```css
.someElement {
	background-image: url("plugin://assets/star.png");
}
```

---

### CSS text-overflow Property Example

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Styles/text-overflow

This CSS snippet demonstrates how to use the 'text-overflow: ellipsis;' property to display an ellipsis when text is clipped. It requires 'overflow: hidden' and 'white-space: nowrap' to function correctly. This property is supported in UXP v3.0 and later.

```css
.someElement {
	overflow: hidden;
	white-space: nowrap;
	text-overflow: ellipsis;
}
```

---

### Manifest Settings for WebView

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

This section details the necessary configurations within your plugin's `manifest.json` file to enable and control the behavior of the WebView component, including domain access and message bridging.

````APIDOC
## Manifest Settings for WebView

### Description

To utilize the `HTMLWebViewElement`, you must configure specific permissions and settings in your plugin's `manifest.json` file. This includes granting the `webview` permission and defining access controls for domains and inter-process communication.

### Method

N/A (This is a configuration file)

### Endpoint

N/A (This is a configuration file)

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```json
{
  "manifestVersion": 5,
  "requiredPermissions": {
    "webview": {
      "allow": "yes",
      "domains": [ "https://*.adobe.com", "https://*.google.com"],
      "enableMessageBridge": "localAndRemote"
    }
  }
}
````

### Response

N/A

#### Success Response (200)

N/A

#### Response Example

N/A

### Manifest Attributes:

- **`allow`**: (Mandatory) Enables WebView access to the plugin. Value: `"yes"`.
- **`allowLocalRendering`**: (Optional) Enables WebView to load local content. Supported from UXP v8.0.0. Value: `"yes"`.
- **`domains`**: (Mandatory) Specifies allowed domains. Supports wildcards (except top-level). Example: `"https://*.adobe.com"`. Can also be `"all"` (not recommended for security and privacy).
- **`enableMessageBridge`**: (Optional) Controls communication between the plugin and WebView content. Possible values:
  - `"localAndRemote"`: Allows communication regardless of content origin (local or remote).
  - `"localOnly"`: Allows communication only for locally loaded content (supported from UXP v8.0.0).
  - `"no"`: Disables communication.

````

--------------------------------

### Attribute Manipulation: get, set, remove, has

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHeadElement

This group of methods provides comprehensive control over an element's attributes. `getAttribute` retrieves an attribute's value, `setAttribute` sets or updates an attribute, `removeAttribute` removes an attribute, and `hasAttribute` checks for the existence of an attribute. `getAttributeNames` returns an array of all attribute names.

```javascript
const element = document.getElementById('myElement');

// Set an attribute
element.setAttribute('data-custom', 'someValue');

// Get an attribute value
const customValue = element.getAttribute('data-custom');
console.log(customValue); // Output: 'someValue'

// Check if an attribute exists
const hasDataCustom = element.hasAttribute('data-custom');
console.log(hasDataCustom); // Output: true

// Remove an attribute
element.removeAttribute('data-custom');

// Get all attribute names
const attributeNames = element.getAttributeNames();
console.log(attributeNames);
````

---

### Set Text Color with CSS in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Styles/color

Demonstrates how to set the text color of an element using CSS within the Adobe Premiere Pro UXP environment. This example uses a named color, but Hex, RGB, RGBA, HSL, and HSLA formats are also supported.

```css
.someElement {
	color: blue;
}
```

---

### Element Attribute Management: getAttribute, setAttribute, removeAttribute, hasAttribute, hasAttributes, getAttributeNames, getAttributeNode, setAttributeNode, removeAttributeNode

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLMenuItemElement

A comprehensive set of methods for managing element attributes. This includes getting attribute values by name, setting new attribute values, removing attributes, checking for the existence of attributes, retrieving all attribute names, and working with attribute nodes. These methods are fundamental for dynamic HTML manipulation.

```javascript
/**
 * Gets the value of the specified attribute on the element.
 * @param {string} name - Name of the attribute whose value you want to get.
 * @returns {string|null} The value of the attribute, or null if it does not exist.
 */
function getAttribute(name) {
	// Implementation details not provided in source
	return null;
}

/**
 * Sets the value of a specified attribute on the element. If the attribute already exists, its value is updated.
 * @param {string} name - Name of the attribute whose value is to be set.
 * @param {string} value - Value to assign to the attribute.
 */
function setAttribute(name, value) {
	// Implementation details not provided in source
}

/**
 * Removes the specified attribute from the element.
 * @param {string} name - Name of the attribute to remove.
 */
function removeAttribute(name) {
	// Implementation details not provided in source
}

/**
 * Returns a boolean value indicating whether the element has the specified attribute.
 * @param {string} name - Name of the attribute to check for.
 * @returns {boolean} True if the attribute exists, false otherwise.
 */
function hasAttribute(name) {
	// Implementation details not provided in source
	return false;
}

/**
 * Returns a boolean value indicating whether the current element has any attributes or not.
 * @returns {boolean} True if the element has any attributes, false otherwise.
 */
function hasAttributes() {
	// Implementation details not provided in source
	return false;
}

/**
 * Returns the attribute names of the element as an Array of strings.
 * @returns {string[]} An array containing the names of all attributes of the element.
 */
function getAttributeNames() {
	// Implementation details not provided in source
	return [];
}

/**
 * Returns the attribute node with the specified name.
 * @param {string} name - The name of the attribute.
 * @returns {Attr|null} The attribute node, or null if it does not exist.
 */
function getAttributeNode(name) {
	// Implementation details not provided in source
	return null;
}

/**
 * Adds or replaces an attribute node with a new attribute node.
 * @param {Attr} newAttr - The attribute node to add or replace.
 * @returns {Attr|null} The old attribute node if it was replaced, otherwise null.
 */
function setAttributeNode(newAttr) {
	// Implementation details not provided in source
	return null;
}

/**
 * Removes an attribute node from the element.
 * @param {Attr} oldAttr - The attribute node to remove.
 * @returns {Attr|null} The removed attribute node, or null if it was not found.
 */
function removeAttributeNode(oldAttr) {
	// Implementation details not provided in source
	return null;
}
```

---

### Create a Generic Entry in a Folder (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/Folder

Shows how to create a new entry (either a file or a folder) within a Folder using the `createEntry` method with specific options for type and overwriting.

```javascript
const myNovel = await aFolder.createEntry("mynovel.txt");
```

```javascript
const catImageCollection = await aFolder.createEntry("cats", { type: types.folder });
```

---

### Event Methods

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Events/Event

This section outlines the methods available on an Event object for initialization and controlling event propagation.

````APIDOC
## initEvent(typeArg, bubblesArg, cancelableArg)

### Description
Initializes an Event object. This method is deprecated and should not be used for new code.

### Method
`initEvent`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **typeArg** (string)
- **bubblesArg** (boolean)
- **cancelableArg** (boolean)

### Request Example
```javascript
// Deprecated usage
// const event = document.createEvent('Event');
// event.initEvent('custom', true, false);
````

### Response

#### Success Response (200)

N/A (Method)

#### Response Example

N/A (Method)

---

## composedPath()

### Description

Returns the event's path.

### Method

`composedPath`

### Parameters

None

### Request Example

```javascript
const path = event.composedPath();
```

### Response

#### Success Response (200)

- **path** (Array) - The event's path.

#### Response Example

```json
[
	{
		"element": "div",
		"id": "container"
	},
	{
		"element": "button"
	}
]
```

---

## preventDefault()

### Description

Cancels the event if it is cancelable, and is not `event.preventDefault()` called.

### Method

`preventDefault`

### Parameters

None

### Request Example

```javascript
if (event.cancelable) {
	event.preventDefault();
}
```

---

## stopImmediatePropagation()

### Description

Stops the propagation of an event to any event listeners and also prevents any further listeners from being called.

### Method

`stopImmediatePropagation`

### Parameters

None

### Request Example

```javascript
event.stopImmediatePropagation();
```

---

## stopPropagation()

### Description

Stops the propagation of an event, in the capture or bubbling phase, but not in the raw event's target.

### Method

`stopPropagation`

### Parameters

None

### Request Example

```javascript
event.stopPropagation();
```

````

--------------------------------

### Render Themed Horizontal Rule with sp-divider (Spectrum UXP)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-html/Hierarchy/hr

Illustrates how to achieve a theme-aware horizontal rule using the Spectrum UXP 'sp-divider' component. This component provides similar rendering to 'hr' but respects themes. Examples show 'large', 'medium', and 'small' sizes.

```HTML
<sp-divider size="large"></sp-divider>
<sp-divider size="medium"></sp-divider>
<sp-divider size="small"></sp-divider>
````

---

### Singleton Modal Dialog Class in JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-modal-dialogs

A JavaScript class implementing the Singleton pattern for modal dialogs. It ensures only one instance exists, encapsulates UI creation, initialization, and execution logic. Dependencies include the host application's DOM. Inputs are optional parameters for dialog initialization, and outputs are the resolved dialog results.

```javascript
class ModalDialog {
	// Singleton support
	static #instance;

	// Private state
	#dialog;
	#params; // (optional) store validated values collected from the UI

	// Enforce the singleton: return the existing instance if present
	constructor() {
		if (ModalDialog.#instance) return ModalDialog.#instance;
		ModalDialog.#instance = this;
	}
	// get (or lazily create) the singleton instance
	static getInstance() {
		if (!ModalDialog.#instance) ModalDialog.#instance = new ModalDialog();
		return ModalDialog.#instance;
	}

	// Build the dialog UI and assign to #dialog
	async createDialog() {
		/* ... */
	}

	// Set defaults and wire listeners
	initDialog() {
		/* ... */
	}

	// Show the dialog and return a Promise with the result
	async runDialog() {
		/* ... */
	}

	// Private: execute Host App DOM logic as needed
	async #runRoutine() {
		/* ... */
	}
}

// Example usage
try {
	const modalDialog = ModalDialog.getInstance();
	await modalDialog.createDialog();
	modalDialog.initDialog();
	const res = await modalDialog.runDialog();
	res;
} catch (error) {
	console.error("Argh!", error);
}
```

---

### Using Spectrum UXP Button Widget

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum

Demonstrates the usage of a built-in Spectrum UXP button widget. This widget mimics the API of the Adobe Spectrum Web Components library and is used directly as an HTML tag.

```html
<sp-button variant="primary">I'm a Spectrum button</sp-button>
```

---

### Comparing Entry Providers (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/Entry

Demonstrates how to compare the `provider` property of two entries to ensure they are serviced by the same file system provider. This is important for operations that require entries from the same source.

```javascript
if (entryOne.provider !== entryTwo.provider) {
	throw new Error("Providers are not the same");
}
```

---

### UXP Manifest for Multiple Panels

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-panels

Configures a UXP plugin manifest (`manifest.json`) to support multiple panels. It defines two distinct panel entrypoints, each with its own label and size configurations, and enables Inter-Plugin Communication (IPC) for cross-panel interactions.

```json
{
	"id": "multi-panel-demo",
	"name": "Multi Panel Demo",
	"version": "1.0.0",
	"main": "index.html",
	"host": { "app": "premierepro", "minVersion": "25.6.0" },
	"manifestVersion": 5,
	"requiredPermissions": {
		// ↓ Required for inter-panel control
		"ipc": { "enablePluginCommunication": true }
	},
	"entrypoints": [
		{
			"id": "uxp-first-panel",
			"type": "panel",
			"label": { "default": "First Panel" },
			"minimumSize": { "width": 430, "height": 500 },
			"preferredDockedSize": { "width": 230, "height": 300 }
			// ...
		},
		{
			"id": "uxp-second-panel",
			"type": "panel",
			"label": { "default": "Second Panel" },
			"minimumSize": { "width": 430, "height": 500 },
			"preferredDockedSize": { "width": 230, "height": 300 }
			// ...
		}
	]
	// ...
}
```

---

### Event Constructor and Properties

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Events/Event

This section details the constructor for creating Event objects and describes the read-only properties available on an Event instance.

````APIDOC
## Event(eventType, eventInit)

### Description
Creates an instance of Event.

### Method
Constructor

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **eventType** (*)
- **eventInit** (*)

### Request Example
```javascript
const myEvent = new Event('customEvent', { detail: { data: 'some data' } });
````

### Response

#### Success Response (200)

N/A (Constructor)

#### Response Example

N/A (Constructor)

---

## Event Properties

### Description

Read-only properties that provide information about the event.

### Properties

- **type** (\*)
- **isTrusted** (boolean)
- **target** (Node)
- **currentTarget** (Node)
- **bubbles** (boolean)
- **cancelable** (boolean)
- **composed** (boolean)
- **eventPhase** (\*)
- **defaultPrevented** (boolean)
- **returnValue** (\*)

### Response Example

```json
{
	"type": "click",
	"isTrusted": true,
	"eventPhase": 2,
	"bubbles": true,
	"cancelable": false,
	"defaultPrevented": false
}
```

````

--------------------------------

### Accessing Plugin Root Folders in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

Provides JavaScript code to access and log the native paths of the plugin, plugin-data, and temporary folders using the UXP localFileSystem API. Requires 'localFileSystem' permission in the manifest.

```javascript
// manifest.json
"requiredPermissions": {
  // permission for localFileSystem
  "localFileSystem": "request"
}

const localFileSystem = require("uxp").storage.localFileSystem;
const pluginFolder = await localFileSystem.getPluginFolder();
const pluginDataFolder = await localFileSystem.getDataFolder();
const tempFolder = await localFileSystem.getTemporaryFolder();


console.log(`pluginFolder = ${pluginFolder.nativePath}`);
console.log(`pluginDataFolder = ${pluginDataFolder.nativePath}`);
console.log(`pluginTempFolder = ${tempFolder.nativePath}`);
````

---

### Dialog-based Debugging in UXP JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/debug

Illustrates the use of `alert()`, `confirm()`, and `prompt()` dialogs for in-UI debugging within UXP plugins. It requires enabling the `enableAlerts` feature flag in the `manifest.json` file. Note that these methods might have limitations in current Premiere Pro versions.

```javascript
// Simple alert dialog
alert("Plugin loaded successfully");

// Confirmation dialog
const confirmed = confirm("Do you want to continue?");
if (confirmed) {
	console.log("User clicked OK");
} else {
	console.log("User clicked Cancel");
}

// Prompt dialog for user input
const userName = prompt("Enter your name:", "Default Name");
console.log(`User entered: ${userName}`);
```

---

### registerNamespace - Register Namespace

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPMeta

Registers a namespace with a suggested prefix. If the prefix is already in use, a new unique prefix is generated and returned.

````APIDOC
## registerNamespace(namespaceURI, suggestedPrefix)

### Description
Registers a namespace with a suggested prefix. If the `suggestedPrefix` is already in use by another namespace, the system will generate, register, and return a different, unique prefix.

### Method
`registerNamespace`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **namespaceURI** (string) - Required - The namespace URI string. Refer to Schema namespace string constants.
* **suggestedPrefix** (string) - Required - The suggested namespace prefix string.

### Request Example
```javascript
XMPMeta.registerNamespace(namespaceURI, suggestedPrefix)
````

### Response

#### Success Response (200)

- **Return Value** (string) - The String containing the actual registered prefix. This will be the `suggestedPrefix` unless that one was already assigned to another namespace, in which case it will be a generated unique prefix.

#### Response Example

```json
{
	"return": "actualPrefix"
}
```

````

--------------------------------

### ReadableStreamDefaultController API Reference

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Streams/ReadableStreamDefaultController

Reference for the ReadableStreamDefaultController interface, providing methods to interact with a readable stream.

```APIDOC
## ReadableStreamDefaultController API

### Description
The `ReadableStreamDefaultController` interface represents the means by which a `ReadableStream` can be controlled. It provides methods to enqueue data, close the stream, and signal errors.

### Methods

#### `desiredSize`

**Description**: Returns the desired size to fill the controlled stream’s internal queue. It can be negative if the queue is over-full.

**Type**: `number`

#### `close()`

**Description**: Closes the controlled readable stream. Consumers will still be able to read any previously-enqueued chunks, but once those are read, the stream will become closed.

**Throws**:
* `TypeError` if the source is not a `ReadableStreamDefaultController`.

#### `enqueue(chunk)`

**Description**: Enqueues the given chunk in the controlled readable stream.

**Parameters**:
* `chunk` (*): The data chunk to enqueue.

**Throws**:
* `TypeError` if the source is not a `ReadableStreamDefaultController`.

#### `error(error)`

**Description**: Errors the controlled readable stream, making all future interactions with it fail with the given error.

**Parameters**:
* `error` (*): The error object to signal.

**Throws**:
* `TypeError` if the source object is not a `ReadableStreamDefaultController`.
````

---

### Accessing Entry Instance via Local File System (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/Entry

Demonstrates how to obtain an Entry instance, specifically a Folder, using the localFileSystem API. This instance can then be used to invoke methods inherited from the Entry base class.

```javascript
const fs = require("uxp").storage.localFileSystem;
const folder = await fs.getPluginFolder(); // returns a Folder instance
const folderEntry = await folder.getEntry("entryName.txt");

// Now we can use folderEntry to invoke the APIs provided by Entry
console.log(folderEntry.isEntry); // isEntry is an API of Entry, in this example it will return true
```

---

### Mixing UI Approaches in UXP Plugins

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/user-interfaces

Demonstrates how to combine standard HTML elements, Spectrum Web Components, and Spectrum UXP Widgets within a single UXP plugin interface. This showcases UXP's flexibility in integrating different UI technologies for a cohesive user experience.

```html
<form>
	<!-- Standard HTML element -->
	<div class="section">
		<!-- Spectrum Web Component -->
		<sp-banner>
			<div slot="header">Welcome</div>
			<div slot="content">This is a mixed UI example</div>
		</sp-banner>

		<!-- Spectrum UXP Widget -->
		<sp-button variant="primary">Submit</sp-button>
	</div>
</form>
```

---

### Shadow DOM API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLDialogElement

API for attaching and managing Shadow DOM.

```APIDOC
## attachShadow(init)

### Description
Attaches a shadow DOM tree to the specified element and returns a reference to its ShadowRoot. This feature is behind a feature flag (`enableSWCSupport`).

### Method
`attachShadow`

### Parameters
#### Path Parameters
- **init** (object) - An object which contains the fields: `mode` (open/closed), `delegatesFocus`, `slotAssignment`.

### Returns
`ShadowRoot`

### See Also
- Element - attachShadow
```

---

### window.fetch()

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/fetch

Fetches a resource from the network. It returns a Promise that resolves to the Response object representing the response to the request.

````APIDOC
## window.fetch(input, [init])

### Description
Fetches a resource from the network. This method is a modern replacement for XMLHttpRequest.

### Method
`fetch`

### Endpoint
`window.fetch(input, [init])`

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Parameters Details
- **input** (`string` or `Request`): Either the URL string to connect with or a `Request` object having the URL and the init option.
- **[init]** (`Object` - Optional): Custom settings for a HTTP request.
  - **[init.method]** (`string`): HTTP request method. The default value is "GET".
  - **[init.headers]** (`Headers`): HTTP request headers to add.
  - **[init.body]** (`string` | `ArrayBuffer` | `TypedArray` | `Blob` | `FormData` | `URLSearchParams`): Body to add to HTTP request.
  - **[init.credentials]** (`string`): Indicates whether to send cookies. The default value is "include". Possible values are "omit" or "include".

### Request Example
```javascript
// Example of a simple GET request
fetch('https://api.example.com/data')
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error('Error:', error));

// Example of a POST request with a JSON body
fetch('https://api.example.com/submit', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ key: 'value' })
})
  .then(response => response.json())
  .then(data => console.log('Success:', data))
  .catch(error => console.error('Error:', error));
````

### Response

#### Success Response (200)

- **Response** (`Promise<Response>`): A Promise that resolves to a `Response` object.

#### Response Example

```json
{
	"status": 200,
	"statusText": "OK",
	"headers": {
		"Content-Type": "application/json"
	},
	"body": "{ \"message\": \"Success\" }"
}
```

### Throws

- `TypeError`: If init.body is set and init.method is either "GET" or "HEAD".
- `TypeError`: If either network error or network time-out occurs after a http request is made.
- `TypeError`: If there is a failure in reading files in FormData during posting FormData.

### Permissions

To leverage `fetch`, update the `manifest.json` with the `network.domains` permission.

```json
{
	"permissions": {
		"network": {
			"domains": ["https://www.adobe.com", "https://*.adobeprerelease.com", "wss://*.myplugin.com"]
		}
	}
}
```

**Limitation:** From UXP v7.4.0 onwards `permissions.network.domains` does not support WildCards in top-level domains.

```json
"domains": [ "https://www.adobe.*", "https://www.*" ] // Invalid example
```

### See Also

- Headers
- Request
- Response

````

--------------------------------

### Shadow DOM and Focus Management

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHeadElement

APIs for attaching shadow DOMs and managing element focus.

```APIDOC
## attachShadow(init)

### Description
Attaches a shadow DOM tree to the specified element and returns a reference to its ShadowRoot. This feature is behind a feature flag `enableSWCSupport`.

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
// const shadowRoot = element.attachShadow({ mode: 'open' });
````

### Response

#### Success Response (200)

Returns a reference to the ShadowRoot.

#### Response Example

N/A

## focus()

### Description

Sets focus on the element.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

N/A

### Request Example

```javascript
// element.focus();
```

### Response

N/A

## blur()

### Description

Removes focus from the element.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

N/A

### Request Example

```javascript
// element.blur();
```

### Response

N/A

````

--------------------------------

### Response Utility and Redirection Methods

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/Response

Utility methods for creating error responses and for redirecting responses.

```APIDOC
### Utility and Redirection Methods

- **clone()**: Creates a copy of the current response object. Returns a `Response`.
- **error()**: Returns a `Response` object representing an error.
- **redirect(url, [status])**: Creates a new response that performs a redirect.
  - **url**: (string) - The URL the new response originates from.
  - **status**: (number) - The status code for the response. Possible values: 301, 302, 303, 307, 308. Default is 302.
  - **Throws**: `RangeError` if the status is not one of the allowed redirect codes.
````

---

### Fetch Data using fetch() in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/network

Demonstrates how to make asynchronous HTTP requests using the modern, promise-based `fetch` API in UXP. It includes error handling for network requests and parsing JSON responses. Network permissions must be declared in `manifest.json`.

```javascript
// Get weather forecast for San Jose
async function getForecast() {
	try {
		const response = await fetch("https://api.weather.gov/gridpoints/MTR/99,82/forecast");

		if (!response.ok) {
			throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
		}
		const data = await response.json();
		console.log(`Forecast: ${data.properties.periods[0].detailedForecast}`);
	} catch (err) {
		console.error("❌ Failed to fetch forecast:", err);
	}
}
```

---

### Create About Dialog Command UI (UXP JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-modal-dialogs

This JavaScript code defines a command for an 'About' dialog in Adobe Premiere Pro UXP. It dynamically creates a dialog element, populates it with version and host information, applies styles programmatically, and displays it as a modal. Dependencies include the 'uxp' package and the local 'manifest.json' file.

```javascript
const { entrypoints, host, versions } = require("uxp");
const manifest = require("./manifest.json");
const os = require("os");

entrypoints.setup({
	commands: {
		"about-command": async () => {
			// Create the dialog dynamically (or load from an HTML file)
			const dialog = document.createElement("dialog");
			dialog.innerHTML = `
        <sp-heading>Clip Mixer</sp-heading>
        <sp-divider size="L"></sp-divider>
        <sp-body>🙌 Thanks for using Clip Mixer v${manifest.version}!</sp-body>
        <sp-body><b>Application:</b> ${host.name} v${host.version} (${os.platform()})</sp-body>
        <sp-body><b>UXP Runtime:</b> ${versions.uxp} - <b>Plugin Version:</b> ${versions.plugin}</sp-body>
      `.trim(); // ☝ trim is a safety measure to avoid whitespace issues

			document.body.appendChild(dialog);

			// Add styles programmatically using element.style
			dialog.style.color = "white";
			dialog.style.padding = "16px";
			dialog.querySelector("sp-divider").style.margin = "0 0 16px 0";
			dialog.querySelector("sp-heading").style.margin = "0 0 16px 0";

			// Show modal
			await dialog.uxpShowModal({
				title: "Command Modal Dialog",
				resize: "none",
				size: { width: 300, height: 200 },
			});
		},
	},
});
```

---

### Check UXP and Premiere Versions Programmatically

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis

This code snippet demonstrates how to retrieve the current host application (Premiere) version and the UXP version within your plugin. It utilizes the 'uxp' module to access this information, which is crucial for implementing version-aware logic.

```javascript
const { host, versions } = require("uxp");
console.log(`Premiere ${host.version}`); // Premiere 25.6.0
console.log(`UXP ${versions.uxp}`); // UXP uxp-8.1.0-local
```

---

### AddTransitionOptions Instance Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/addtransitionoptions

This section details the instance methods available for the AddTransitionOptions class, used to modify its properties.

```APIDOC
## AddTransitionOptions Instance Methods

### Description
Methods to set the various properties of the AddTransitionOptions class.

### Methods
#### setApplyToStart

- **Description**: Sets whether to apply the transition to the start or end of the track item.
- **Parameters**:
  - **applyToStart** (boolean) - Description: N/A

#### setDuration

- **Description**: Sets the duration of the transition.
- **Parameters**:
  - **tickTime** (TickTime) - Description: Sets the duration of the transition in TickTime.

#### setForceSingleSided

- **Description**: Sets whether the transition should be applied to one or both sides.
- **Parameters**:
  - **forceSingleSided** (boolean) - Description: N/A

#### setTransitionAlignment

- **Description**: Sets the alignment of the transition.
- **Parameters**:
  - **transitionAlignment** (number) - Description: N/A
```

---
