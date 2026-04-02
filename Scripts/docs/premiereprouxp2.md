## getAttributeNode(name)

### Description

Returns the attribute node with the specified name.

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
// const idAttributeNode = element.getAttributeNode('id');
```

### Response

#### Success Response (200)

- **Attribute Node** (`*`) - The attribute node object, or `null` if not found.

#### Response Example

[Conceptual example, actual object structure depends on attribute node implementation]

```json
{
	"//": "Represents an AttributeNode object"
}
```

## setAttributeNode(newAttr)

### Description

Adds or replaces an attribute node in the element.

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
// const newIdNode = document.createAttribute('id');
// newIdNode.value = 'new-id';
// element.setAttributeNode(newIdNode);
```

### Response

#### Success Response (200)

[No explicit return value, behavior is side-effectual]

#### Response Example

[None]

## removeAttributeNode(oldAttr)

### Description

Removes an attribute node from the element.

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
// const idNodeToRemove = element.getAttributeNode('id');
// if (idNodeToRemove) {
//   element.removeAttributeNode(idNodeToRemove);
// }
```

### Response

#### Success Response (200)

[No explicit return value, behavior is side-effectual]

#### Response Example

[None]

````

--------------------------------

### Response Constructor and Properties

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/Response

Details on how to construct a Response object and an overview of its key properties.

```APIDOC
## Response Class

Represents a resource request.

**Since**: v7.3.0

### Constructor

`Response([body], [options])`

- **body**: (string | Blob | ArrayBuffer | TypedArray | FormData | ReadableStream | URLSearchParams | null) - The body of the response.
- **options**: (Object) - Custom settings for the response.
  - **status**: (number) - The status code of the response. Default is 200.
  - **statusText**: (string) - The status message associated with the status code. Default is "".
  - **headers**: (Headers | string | {}) - Headers to add to the response.

### Properties

- **body**: (ReadableStream | null) - Read-only. ReadableStream object with the body contents or null if the response's body is empty.
- **bodyUsed**: (boolean) - Read-only. Indicates whether the response body has been read yet.
- **headers**: (Headers) - Read-only. Headers object associated with the response.
- **ok**: (boolean) - Read-only. Indicates whether the response was successful (status in range 200-299).
- **status**: (number) - Read-only. HTTP status code of the response.
- **statusText**: (string) - Read-only. Status message corresponding to the HTTP status code.
- **url**: (string) - Read-only. URL of the response.
````

---

### Implement UXP Command Handlers

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/entrypoints

Sets up command handlers for UXP plugins. These functions are invoked when a command entrypoint is activated. Handlers are defined in the `commands` property of `entrypoints.setup()` and must correspond to command IDs in `manifest.json`. The handler receives a `uxpcommand` event object.

```javascript
const { entrypoints } = require("uxp");

const commandHandler = (evt) => {
	console.log(
		"Command handler invoked!",
		evt.type, // uxpcommand
	);
};

entrypoints.setup({
	commands: {
		firstCommand: commandHandler, // 👈 must match the id of the
		// entrypoint of type "command"
		// from manifest.json
	},
});
```

---

### CSS border-style Example

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Styles/border-style

Demonstrates how to apply a solid white border to an element using CSS in UXP. This snippet requires no external dependencies and results in a styled button element.

```css
.button {
	border-width: 2px;
	border-style: solid;
	border-color: white;
}
```

---

### Launch External URL with Clear Context (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Demonstrates how to use `shell.openExternal` to open a URL in the user's browser. It emphasizes providing clear, user-friendly text in the `developerText` parameter for the consent dialog. This is the recommended approach for opening external web resources.

```javascript
await shell.openExternal("https://example.com/guide", "Opening tutorial guide in your browser");
```

---

### Launch Applications via URL Schemes with UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Use `shell.openExternal()` to launch external applications using URL schemes. This is useful for opening websites, composing emails, or triggering platform-specific apps. Requires `launchProcess` permission with specified schemes in `manifest.json`. Returns an empty string on success or an error message on failure.

```javascript
const { shell } = require("uxp");

// Open a website in the default browser 🌐
async function openDocumentation() {
	try {
		const result = await shell.openExternal("https://developer.adobe.com/", "Opening Adobe Developer documentation");

		if (result === "") {
			console.log("✅ Browser opened successfully");
		} else {
			console.error(`❌ Failed to open browser: ${result}`);
		}
	} catch (err) {
		console.error("Error opening browser:", err);
	}
}

// Compose an email with pre-filled content 📧
async function sendFeedbackEmail() {
	try {
		const subject = encodeURIComponent("Plugin Feedback");
		const body = encodeURIComponent("I have feedback about your plugin...");

		const result = await shell.openExternal(`mailto:support@example.com?subject=${subject}&body=${body}`, "Opening mail client to send feedback");

		if (result === "") {
			console.log("✅ Mail client opened successfully");
		} else {
			console.error(`❌ Failed to open mail client: ${result}`);
		}
	} catch (err) {
		console.error("Error opening mail client:", err);
	}
}

// Open Maps to a specific location 🗺️
async function openLocationInMaps() {
	try {
		// For macOS: use maps:// scheme
		const macResult = await shell.openExternal("maps://?address=345+Park+Ave+San+Jose", "Opening Maps to Adobe office location");

		// For Windows: use bingmaps: scheme
		// const winResult = await shell.openExternal(
		//   "bingmaps:?q=345+Park+Ave+San+Jose,+95110",
		//   "Opening Maps to Adobe office location"
		// );

		if (macResult === "") {
			console.log("✅ Maps opened successfully");
		} else {
			console.error(`❌ Failed to open Maps: ${macResult}`);
		}
	} catch (err) {
		console.error("Error opening Maps:", err);
	}
}
```

---

### Handle User Denial When Opening Path (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Illustrates how to use `shell.openPath` to open a local file and gracefully handle cases where the user denies the request or the operation fails. It checks the return value of `openPath` to determine if the action was successful and logs a message to the console if it was not.

```javascript
const result = await shell.openPath(filePath, "Opening project file");
if (result !== "") {
	// User denied or operation failed
	console.log("Unable to open file. Please open it manually.");
}
```

---

### Modal Dialog Creation and Initialization (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-modal-dialogs

This snippet demonstrates the instantiation and initialization of a `ModalDialog` object. It includes creating the dialog, setting up event listeners, and attaching necessary data attributes. It relies on a `ModalDialog.getInstance()` method and `uxpShowModal` for displaying the dialog.

```javascript
      // Mark the dialog element itself as having listeners attached
      this.#dialog.dataset.listenersAttached = "true";
    }
  }


  async runDialog() {
    const rv = await this.#dialog.uxpShowModal({
      title: G.title,
      resize: "none",
      size: G.dialogSize,
    });


    if (rv === "ok") {
      console.log("Dialog closed with OK");
      const res = await this.#runRoutine();
      if (res === true) return true;
      throw res; // bubble any routine error
    }


    if (rv === "cancel" || rv === "reasonCanceled") {
      throw "cancel";
    }


    throw "Dialog closed unexpectedly";
  }


  // Run whatever Host App DOM code is needed
  // Has access to validated params via this.#params
  async #runRoutine() {
    console.log("Running PPro routine with params:", this.#params);
    //... perform the (fictitious) routine using
    // this.#params.width and this.#params.height
    return true;
  }


  // Getter to access validated params (if needed by external code)
  getParams() {
    // Return a copy to prevent external mutation
    return { ...this.#params };
  }
}


const btn = document.querySelector("#openDialogBtn");
if (btn && !btn.dataset.listenerAttached) {
  btn.addEventListener("click", async () => {
    try {
      // const modalDialog = new ModalDialog() // 👈 same 👇
      const modalDialog = ModalDialog.getInstance();
      await modalDialog.createDialog();
      modalDialog.initDialog();
      const res = await modalDialog.runDialog();
      res;
    } catch (error) {
      console.error("Arrgh!", error);
    }
  });
  // ✅ ensure we only wire the listener once
  btn.dataset.listenerAttached = "true";
}
```

---

### Manage XMP Metadata Properties using XMPMeta

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPMeta

Demonstrates creating an XMPMeta object and using property-based APIs to set, get, and delete metadata properties. It also shows how to check for property existence. This requires the 'uxp' module.

```javascript
let { XMPMeta, XMPConst } = require("uxp").xmp;
let meta = new XMPMeta();
meta.setProperty(XMPConst.NS_XMP, "Name", "vkumarg");
let prop = meta.getProperty(XMPConst.NS_XMP, "Name");
console.log(prop.namespace);
console.log(prop.options);
console.log(prop.path);
// checking for the property existence and deleting it
if (meta.doesPropertyExist(XMPConst.NS_XMP, "Name")) {
	meta.deleteProperty(XMPConst.NS_XMP, "Name");
}

if (!meta.doesPropertyExist(XMPConst.NS_XMP, "Name")) {
	console.log("Property doesn't exist");
} else {
	console.log("Property exists");
}
```

---

### Configure Single Host Application for UXP Plugin (Production)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/distribution/package

This JSON configuration specifies a single host application and its minimum version for a UXP plugin. It is suitable for production deployments.

```json
"host": {
  "app": "premierepro", "minVersion": "25.6.0"
}
```

---

### Attribute Management API (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLinkElement

Functions for getting, setting, and removing attributes on an element. Supports retrieving all attribute names and attribute nodes.

```javascript
/**
 * Returns the value of the specified attribute on the element.
 * @param {string} name - Name of the attribute whose value you want to get.
 * @returns {string} The value of the attribute.
 */
function getAttribute(name) {
	// Implementation details...
}

/**
 * Sets the value of the specified attribute on the element.
 * @param {string} name - Name of the attribute whose value is to be set.
 * @param {string} value - Value to assign to the attribute.
 */
function setAttribute(name, value) {
	// Implementation details...
}

/**
 * Removes the specified attribute from the element.
 * @param {string} name - Name of the attribute to remove.
 */
function removeAttribute(name) {
	// Implementation details...
}

/**
 * Returns a boolean value indicating whether the current element has the specified attribute.
 * @param {string} name - Name of the attribute to check for.
 * @returns {boolean} True if the attribute exists, false otherwise.
 */
function hasAttribute(name) {
	// Implementation details...
}

/**
 * Returns a boolean value indicating whether the current element has any attributes or not.
 * @returns {boolean} True if the element has any attributes, false otherwise.
 */
function hasAttributes() {
	// Implementation details...
}

/**
 * Returns the attribute names of the element as an Array of strings.
 * @returns {Array<string>} An array containing the names of all attributes.
 */
function getAttributeNames() {
	// Implementation details...
}

/**
 * Returns the attribute node with the specified name.
 * @param {string} name - The name of the attribute node to retrieve.
 * @returns {*} The attribute node.
 */
function getAttributeNode(name) {
	// Implementation details...
}

/**
 * Adds a new attribute node to the element.
 * @param {*} newAttr - The attribute node to add.
 * @returns {*} The added attribute node.
 */
function setAttributeNode(newAttr) {
	// Implementation details...
}

/**
 * Removes an attribute node from the element.
 * @param {*} oldAttr - The attribute node to remove.
 * @returns {*} The removed attribute node.
 */
function removeAttributeNode(oldAttr) {
	// Implementation details...
}
```

---

### Get XMPFileInfo Object using XMPFile

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPFile

Shows how to use the XMPFile class to retrieve XMPFileInfo, which includes the file path and format. This is achieved by instantiating XMPFile and then calling the getFileInfo() method.

```javascript
const { XMPFile } = require("uxp").xmp;

// Create a new XMPFile object
const xmpFile = new XMPFile("sample.psd", XMPConst.FILE_PHOTOSHOP, XMPConst.OPEN_FOR_UPDATE);

// Get XMPFileInfo object
const xmpFileInfo = xmpFile.getFileInfo();
console.log(xmpFileInfo.filePath);
console.log(xmpFileInfo.format);
```

---

### getNamespaceURI - Get Namespace URI

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPMeta

Retrieves the registered namespace URI associated with a given namespace prefix.

````APIDOC
## getNamespaceURI(namespacePrefix)

### Description
Retrieves the registered namespace URI associated with a namespace prefix.

### Method
`getNamespaceURI`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **namespacePrefix** (string) - Required - The namespace prefix string.

### Request Example
```javascript
XMPMeta.getNamespaceURI(namespacePrefix)
````

### Response

#### Success Response (200)

- **Return Value** (string) - The URI string associated with the prefix.

#### Response Example

```json
{
	"return": "http://example.com/uri"
}
```

````

--------------------------------

### Element Methods

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLinkElement

This section covers various methods applicable to HTML Elements, such as checking for pointer capture, getting bounding client rectangles, matching selectors, and manipulating adjacent content.

```APIDOC
## hasPointerCapture(pointerId)

### Description
Checks if the element has pointer capture for the specified pointer.

### Method
GET

### Endpoint
Element.hasPointerCapture

### Parameters
#### Path Parameters
- **pointerId** (`number`) - Required - The unique identifier of the pointer to check.

### Response
#### Success Response (200)
- **boolean** - True if the element has pointer capture for the specified pointer, false otherwise.

### Response Example
```json
true
````

````

```APIDOC
## getBoundingClientRect()

### Description
Returns the size of an element and its position relative to the viewport.

### Method
GET

### Endpoint
Element.getBoundingClientRect

### Parameters
No parameters.

### Response
#### Success Response (200)
- *** (DOMRect)** - An object with `top`, `right`, `bottom`, `left`, `width`, and `height` properties.

### Response Example
```json
{
  "top": 10,
  "right": 50,
  "bottom": 60,
  "left": 10,
  "width": 40,
  "height": 50
}
````

````

```APIDOC
## closest(selectorString)

### Description
Tests the element, and recursively its parent elements, for a match against the provided CSS selector. It returns the first ancestor element that matches the selector, or null if none match.

### Method
GET

### Endpoint
Element.closest

### Parameters
#### Query Parameters
- **selectorString** (`string`) - Required - The CSS selector to match.

### Response
#### Success Response (200)
- **Element** - The first ancestor element that matches the selector, or null if none match.

### Response Example
```json
{
  "tagName": "div",
  "id": "matchingElement"
}
````

````

```APIDOC
## matches(selectorString)

### Description
Checks if the element matches the provided CSS selector.

### Method
GET

### Endpoint
Element.matches

### Parameters
#### Query Parameters
- **selectorString** (`string`) - Required - The CSS selector to match.

### Response
#### Success Response (200)
- **boolean** - True if the element matches the selector, false otherwise.

### Response Example
```json
true
````

````

```APIDOC
## insertAdjacentHTML(position, value)

### Description
Inserts an HTML string into the DOM at the specified position relative to the element.

### Method
POST

### Endpoint
Element.insertAdjacentHTML

### Parameters
#### Request Body
- **position** (`string`) - Required - The position to insert the HTML. Can be 'beforebegin', 'afterbegin', 'beforeend', or 'afterend'.
- **value** (`string`) - Required - The HTML string to insert.

### Request Example
```json
{
  "position": "afterend",
  "value": "<p>New paragraph</p>"
}
````

### Response

#### Success Response (200)

No content returned on success.

````

```APIDOC
## insertAdjacentElement(position, node)

### Description
Inserts a DOM `Node` into the DOM at the specified position relative to the element.

### Method
POST

### Endpoint
Element.insertAdjacentElement

### Parameters
#### Request Body
- **position** (`string`) - Required - The position to insert the element. Can be 'beforebegin', 'afterbegin', 'beforeend', or 'afterend'.
- **node** (`Node`) - Required - The `Node` to insert.

### Request Example
```json
{
  "position": "beforeend",
  "node": "<div id=\"newElement\"></div>"
}
````

### Response

#### Success Response (200)

- **Node** - The inserted `Node`.

### Response Example

```json
{
	"tagName": "div",
	"id": "newElement"
}
```

````

```APIDOC
## insertAdjacentText(position, text)

### Description
Inserts a text string into the DOM at the specified position relative to the element.

### Method
POST

### Endpoint
Element.insertAdjacentText

### Parameters
#### Request Body
- **position** (`string`) - Required - The position to insert the text. Can be 'beforebegin', 'afterbegin', 'beforeend', or 'afterend'.
- **text** (`string`) - Required - The text string to insert.

### Request Example
```json
{
  "position": "afterbegin",
  "text": "Inserted text."
}
````

### Response

#### Success Response (200)

No content returned on success.

````

```APIDOC
## hasChildNodes()

### Description
Checks if the element has any child nodes.

### Method
GET

### Endpoint
Element.hasChildNodes

### Parameters
No parameters.

### Response
#### Success Response (200)
- **boolean** - True if the element has child nodes, false otherwise.

### Response Example
```json
true
````

````

```APIDOC
## cloneNode(deep)

### Description
Creates a copy of the element. If `deep` is true, all descendants are copied as well.

### Method
GET

### Endpoint
Element.cloneNode

### Parameters
#### Query Parameters
- **deep** (`boolean`) - Optional - If true, clones all descendants. Defaults to false.

### Response
#### Success Response (200)
- **Node** - A clone of the element.

### Response Example
```json
{
  "tagName": "div",
  "textContent": "Cloned content"
}
````

````

```APIDOC
## appendChild(child)

### Description
Appends a node to the end of the list of children of a specified parent node. If the node being inserted is already in the document, it is first removed from its current parent.

### Method
POST

### Endpoint
Element.appendChild

### Parameters
#### Request Body
- **child** (`Node`) - Required - The node to append.

### Request Example
```json
{
  "child": "<p>Appended child</p>"
}
````

### Response

#### Success Response (200)

- **Node** - The appended `Node`.

### Response Example

```json
{
	"tagName": "p",
	"textContent": "Appended child"
}
```

````

```APIDOC
## insertBefore(child, before)

### Description
Inserts a node before a specified reference node as a child of the given parent node. If the reference node is null, the value is inserted at the end of the list of children.

### Method
POST

### Endpoint
Element.insertBefore

### Parameters
#### Request Body
- **child** (`Node`) - Required - The node to insert.
- **before** (`Node`) - Optional - The reference node before which the new node is to be inserted.

### Request Example
```json
{
  "child": "<span>Inserted node</span>",
  "before": "<p>Existing node</p>"
}
````

### Response

#### Success Response (200)

- **Node** - The inserted `Node`.

### Response Example

```json
{
	"tagName": "span",
	"textContent": "Inserted node"
}
```

````

```APIDOC
## replaceChild(newChild, oldChild)

### Description
Replaces a child `Node` in the given parent node with a new node.

### Method
POST

### Endpoint
Element.replaceChild

### Parameters
#### Request Body
- **newChild** (`Node`) - Required - The new node to replace the old one with.
- **oldChild** (`Node`) - Required - The node to be replaced.

### Request Example
```json
{
  "newChild": "<p>Replaced content</p>",
  "oldChild": "<span>Old content</span>"
}
````

### Response

#### Success Response (200)

- **Node** - The node that was replaced (`oldChild`).

### Response Example

```json
{
	"tagName": "span",
	"textContent": "Old content"
}
```

````

```APIDOC
## removeChild(child)

### Description
Removes a child `Node` from the given parent node.

### Method
DELETE

### Endpoint
Element.removeChild

### Parameters
#### Request Body
- **child** (`Node`) - Required - The child node to remove.

### Request Example
```json
{
  "child": "<p>Node to remove</p>"
}
````

### Response

#### Success Response (200)

- **Node** - The removed `Node`.

### Response Example

```json
{
	"tagName": "p",
	"textContent": "Node to remove"
}
```

````

```APIDOC
## remove()

### Description
Removes the element from its parent.

### Method
DELETE

### Endpoint
Element.remove

### Parameters
No parameters.

### Response
#### Success Response (200)
No content returned on success.

````

````APIDOC
## before(...nodes)

### Description
Inserts one or more `Node` objects or `DOMString`s into the child list of a parent, inserted immediately before the parent's first child. This is a sibling of the parent.

### Method
POST

### Endpoint
Element.before

### Parameters
#### Request Body
- **...nodes** (`Array<Node>` or `Array<string>`) - Required - The nodes or strings to insert.

### Request Example
```json
{
  "nodes": [
    "<p>Sibling 1</p>",
    document.createElement('div')
  ]
}
````

### Response

#### Success Response (200)

No content returned on success.

````

```APIDOC
## after(...nodes)

### Description
Inserts one or more `Node` objects or `DOMString`s into the child list of a parent, inserted immediately after the parent's last child. This is a sibling of the parent.

### Method
POST

### Endpoint
Element.after

### Parameters
#### Request Body
- **...nodes** (`Array<Node>` or `Array<string>`) - Required - The nodes or strings to insert.

### Request Example
```json
{
  "nodes": [
    "<p>Sibling 1</p>",
    document.createElement('div')
  ]
}
````

### Response

#### Success Response (200)

No content returned on success.

````

```APIDOC
## replaceWith(...nodes)

### Description
Replaces the element with one or more `Node` objects or `DOMString`s.

### Method
POST

### Endpoint
Element.replaceWith

### Parameters
#### Request Body
- **...nodes** (`Array<Node>` or `Array<string>`) - Required - The nodes or strings to replace the element with.

### Request Example
```json
{
  "nodes": [
    "<p>Replaced element</p>",
    document.createElement('span')
  ]
}
````

### Response

#### Success Response (200)

No content returned on success.

````

```APIDOC
## contains(node)

### Description
Checks if the element contains the given node.

### Method
GET

### Endpoint
Element.contains

### Parameters
#### Query Parameters
- **node** (`Node`) - Required - The node to check for.

### Response
#### Success Response (200)
- **boolean** - True if the element contains the node, false otherwise.

### Response Example
```json
true
````

````

```APIDOC
## getRootNode(options)

### Description
Returns the root node of the element. This is usually the document, but can be a shadow root if the element is inside a shadow DOM.

### Method
GET

### Endpoint
Element.getRootNode

### Parameters
#### Query Parameters
- **options** (`Object`) - Optional - Configuration options. Currently supports `composed` (boolean).

### Response
#### Success Response (200)
- **Node** - The root node of the element.

### Response Example
```json
{
  "nodeType": 9, // Node.DOCUMENT_NODE
  "nodeName": "#document"
}
````

````

--------------------------------

### Configure Manifest for Launching Processes

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Declare the `launchProcess` permission in `manifest.json` to allow your plugin to open files and launch applications. Specify allowed file extensions and URL schemes. An empty string in extensions permits opening folders.

```json
{
  "manifestVersion": 5,
  "requiredPermissions": {
    "launchProcess": {
      "extensions": [".pdf", ".txt", ".mp4", ""],
      "schemes": ["https", "mailto"]
    }
  }
}
````

---

### HTML Head Element in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-html/General/head

Specifies metadata for UXP HTML documents, including styles and scripts. This example demonstrates basic CSS styling within the head tag to highlight text.

```html
<!DOCTYPE html>
<html>
	<head>
		<style>
			.highlight {
				color: red;
			}
		</style>
	</head>
	<body>
		<div class="highlight">Hello, world</div>
	</body>
</html>
```

---

### UXP Manifest - Requesting Local File System Permission

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Specifies the required permissions for a UXP plugin in its manifest file. In this case, it requests 'request' level access to the local file system, enabling interactive file operations.

```json
{
	"manifestVersion": 5,
	// ...
	"requiredPermissions": {
		"localFileSystem": "request"
	}
	// ...
}
```

---

### Pointer Capture Example in JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLElement

Demonstrates how to use `setPointerCapture` and `releasePointerCapture` to manage pointer events for dragging elements. This allows an element to continuously receive pointer events even if the pointer leaves its bounds.

```javascript
// HTML
// <style>
//     div {
//         width: 140px;
//         height: 50px;
//         display: flex;
//         align-items: center;
//         justify-content: center;
//         background: #fbe;
//         position: absolute;
//     }
// </style>
// <div id="slider">SLIDE ME</div>

// JS
const slider = document.getElementById("slider");

function beginSliding(e) {
	slider.setPointerCapture(e.pointerId);
	slider.addEventListener("pointermove", slide);
}

function stopSliding(e) {
	slider.releasePointerCapture(e.pointerId);
	slider.removeEventListener("pointermove", slide);
}

function slide(e) {
	slider.style.left = e.clientX + "px"; // Ensure units are appended
}

slider.addEventListener("pointerdown", beginSliding);
slider.addEventListener("pointerup", stopSliding);
```

---

### Get All Attribute Names (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLTextAreaElement

Returns an array containing the names of all attributes present on the element. Compatible with the standard DOM API.

```javascript
/**
 * Returns the attribute names of the element as an Array of strings.
 * @returns {Array<string>} An array of attribute names.
 */
Element.prototype.getAttributeNames();
```

---

### Scaffolded UXP Plugin File Structure

Source: https://developer.adobe.com/premiere-pro/uxp/plugins

This structure represents the basic files generated when scaffolding a new UXP plugin using a starter template. It includes manifest for configuration, HTML for UI, JS for logic, and README for documentation.

```text
Test-fm0dom
├── manifest.json     🔧 Plugin configuration
├── index.html        🌐 User Interface
├── index.js          💻 Logic
└── README.md         📝 Documentation
```

---

### Get All Element Attribute Names

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLStyleElement

Returns an array containing the names of all attributes present on the element. This can be useful for iterating over all attributes.

```javascript
const element = document.getElementById("myElement");
const attributeNames = element.getAttributeNames();
console.log(attributeNames);
```

---

### Get Element Attribute Value (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLabelElement

Retrieves the value of a specified attribute from an element. Returns null if the attribute is not set.

```javascript
/**
 * Returns the value of the specified attribute on the element.
 *
 * @param {string} name - Name of the attribute whose value you want to get.
 * @returns {string|null} The value of the attribute, or null if it's not set.
 */
function getAttribute(name) {
	// Implementation details for getting attribute
}
```

---

### FolderItem Instance Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/folderitem

Instance methods for FolderItem, including actions for creating bins, moving items, and renaming.

```APIDOC
## FolderItem Instance Methods

### Description
Instance methods for FolderItem objects.

### Method: createBinAction
#### Description
Returns an action that lets users create a new bin.
#### Parameters
- **name** (_string_) - Description: -
- **makeUnique** (_boolean_) - Description: -
```

```APIDOC
### Method: createMoveItemAction
#### Description
Creates an action that moves the given item to the provided folder item `newParent`.
#### Parameters
- **item** (_ProjectItem_) - Description: -
- **newParent** (_FolderItem_) - Description: -
```

```APIDOC
### Method: createRemoveItemAction
#### Description
Creates an action that removes the given item from this folder.
#### Parameters
- **item** (_ProjectItem_) - Description: -
```

```APIDOC
### Method: createRenameBinAction
#### Description
Renames the Bin and returns true if it's successful.
#### Parameters
- **name** (_string_) - Description: -
```

```APIDOC
### Method: createSetColorLabelAction
#### Description
Creates an action for setting the color label of a projectItem by index.
#### Parameters
- **inColorLabelIndex** (_number_) - Description: -
```

```APIDOC
### Method: createSetNameAction
#### Description
Returns an action that renames a projectItem.
#### Parameters
- **inName** (_string_) - Description: -
```

```APIDOC
### Method: createSmartBinAction
#### Description
Creates a smart bin with the given name and returns the Folder object.
#### Parameters
- **name** (_string_) - Description: -
- **searchQuery** (_string_) - Description: -
```

```APIDOC
### Method: getColorLabelIndex
#### Description
Get the color label index of the projectItem.
#### Returns
- (_number_)
```

```APIDOC
### Method: getItems
#### Description
Collection of child items of this folder.
#### Returns
- (_ProjectItem[]_)
```

```APIDOC
### Method: getProject
#### Description
Get the parent Project of this projectItem.
#### Returns
- (_Project_)
```

---

### Save to User-Selected Location with UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Opens a save file dialog, allowing the user to specify a filename and location to save a text file. It then writes predefined content to the selected file. Requires 'request' level permission for localFileSystem.

```javascript
const { localFileSystem, domains, fileTypes } = require("uxp").storage;

async function saveToUserSelectedLocation() {
	// Present a save dialog to let the user choose where to save
	try {
		const file = await localFileSystem.getFileForSaving("export.txt", {
			types: ["txt"],
		});

		if (!file) {
			console.log("User cancelled save operation");
			return;
		}

		// Write content to the selected location
		await file.write("This content was exported from the plugin.");
		console.log(`File saved to: ${file.nativePath}`);
	} catch (err) {
		console.error("Failed to save file:", err);
	}
}
```

---

### ResizeObserverEntry Constructor

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20DOM/ResizeObserverEntry

Initializes a ResizeObserverEntry object, which is passed to the ResizeObserver callback function.

````APIDOC
## ResizeObserverEntry(target)

### Description
Represents the object passed to the ResizeObserver() constructor's callback function, which allows access to the new dimensions of the Element.

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
new ResizeObserverEntry(targetElement);
````

### Response

#### Success Response (N/A)

N/A

#### Response Example

N/A

````

--------------------------------

### Moving an Entry to a Folder (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/Entry

Demonstrates the `moveTo` method for relocating an entry to a different folder. Examples cover basic moving, moving with overwrite, and moving with a new name.

```javascript
await someFile.moveTo(someFolder);
````

```javascript
await someFile.moveTo(someFolder, { overwrite: true });
```

```javascript
await someFolder.moveTo(anotherFolder, { overwrite: true });
```

```javascript
await someFile.moveTo(someFolder, { newName: "masterpiece.txt" });
```

```javascript
await someFile.moveTo(someFolder, {newName: 'novel.txt', {overwrite: true});
```

---

### Get Elements by Class Name

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLStyleElement

Returns a live `NodeList` of all descendant elements that have the specified class name. This method can be used to select multiple elements.

```javascript
const listItems = document.getElementsByClassName("list-item");
for (let i = 0; i < listItems.length; i++) {
	listItems[i].style.color = "blue";
}
```

---

### Spectrum UXP Widgets for Native Look

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/user-interfaces

Showcases the use of built-in Spectrum UXP Widgets for creating UI components that match Premiere's native look and feel. These widgets require no external installation or imports, simplifying rapid prototyping.

```html
<sp-button variant="primary">I'm a Spectrum button</sp-button>
<sp-textfield placeholder="Enter your name"></sp-textfield>
<sp-checkbox>Enable feature</sp-checkbox>
```

---

### UxpMenuItem Class Reference

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Entry%20Points/UxpMenuItem

Provides details on the properties and methods available for a UxpMenuItem object.

```APIDOC
## UxpMenuItem

Class describing a single menu item of a panel.

### Properties

- **id** (string) - Read only. Get menu item id.
- **label** (string) - Get menu item label, localized string.
- **enabled** (boolean) - Get menu item enable state.
- **checked** (boolean) - Get menu item checked state.
- **submenu** (UxpMenuItems) - Get menu submenu.
- **parent** (UxpMenuItems) - Get menu parent.

### Methods

- **UxpMenuItem()**
  * Constructor for UxpMenuItem.

- **label** (string label)
  * Set label of the menu item. The label will be updated immediately, asynchronously.
  * Param: `label` (string) - should be a localized string.

- **enabled** (boolean enabled)
  * Set enabled state of the menu item. The state will be updated immediately, asynchronously.
  * Param: `enabled` (boolean).

- **checked** (boolean checked)
  * Set checked state of the menu item. The state will be updated immediately, asynchronously.
  * Param: `checked` (boolean).

- **remove()**
  * Remove the menu item.
```

---

### Attribute Manipulation (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLBodyElement

Provides methods to get, set, remove, and check for the existence of element attributes. Supports retrieving attribute names and nodes.

```javascript
/**
 * Gets the value of the specified attribute.
 * @param {string} name - Name of the attribute whose value you want to get.
 * @returns {string} The value of the attribute.
 */
function getAttribute(name) {
	// Implementation details would go here
}

/**
 * Sets the value of a specified attribute on the element.
 * @param {string} name - Name of the attribute whose value is to be set.
 * @param {string} value - Value to assign to the attribute.
 */
function setAttribute(name, value) {
	// Implementation details would go here
}

/**
 * Removes the specified attribute from the element.
 * @param {string} name - Name of the attribute to remove.
 */
function removeAttribute(name) {
	// Implementation details would go here
}

/**
 * Checks if the element has the specified attribute.
 * @param {string} name - The name of the attribute to check for.
 * @returns {boolean} True if the attribute exists, false otherwise.
 */
function hasAttribute(name) {
	// Implementation details would go here
}

/**
 * Returns a boolean value indicating whether the current element has any attributes or not.
 * @returns {boolean} True if the element has any attributes, false otherwise.
 */
function hasAttributes() {
	// Implementation details would go here
}

/**
 * Returns the attribute names of the element as an Array of strings.
 * @returns {Array<string>} An array containing the names of all attributes.
 */
function getAttributeNames() {
	// Implementation details would go here
}

/**
 * Returns the specified attribute node.
 * @param {string} name - The name of the attribute.
 * @returns {*} The attribute node.
 */
function getAttributeNode(name) {
	// Implementation details would go here
}

/**
 * Sets the specified attribute node to the element.
 * @param {*} newAttr - The attribute node to set.
 */
function setAttributeNode(newAttr) {
	// Implementation details would go here
}

/**
 * Removes the specified attribute node from the element.
 * @param {*} oldAttr - The attribute node to remove.
 */
function removeAttributeNode(oldAttr) {
	// Implementation details would go here
}
```

---

### Configure UXP Manifest for Launching Applications

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

The `manifest.json` file specifies the URL schemes that your UXP plugin is allowed to launch using `launchProcess`. Ensure that all schemes used in `shell.openExternal()` or `shell.openPath()` are listed under `requiredPermissions.launchProcess.schemes`.

```json
{
	"manifestVersion": 5,
	// ...
	"requiredPermissions": {
		"launchProcess": {
			"schemes": ["https", "mailto", "maps", "bingmaps"]
		}
	}
	// ...
}
```

---

### TypeScript Example: Analyze Premiere Pro Sequence

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/typescript-support

Demonstrates importing types, accessing the active project and sequence, and iterating through video tracks to log track names. Features TypeScript's type inference and safety.

```typescript
// Import types from types.d.ts in the root directory
import type { premierepro, Sequence, VideoTrack } from "../types";

const ppro = require("premierepro") as premierepro;

async function analyzeSequence(): Promise<void> {
	const project = await ppro.Project.getActiveProject();
	if (!project) return;

	// TypeScript infers types automatically
	const sequence = await project.getActiveSequence();
	if (!sequence) return;

	// Full type safety and IntelliSense
	const videoTrackCount: number = await sequence.getVideoTrackCount();

	for (let i = 0; i < videoTrackCount; i++) {
		const track: VideoTrack = await sequence.getVideoTrack(i);
		console.log(`Track ${i}: ${track.name}`);
	}
}
```

---

### Get Attribute Node (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLScriptElement

Retrieves the attribute node for a specified attribute name. This returns a node object representing the attribute, which can be manipulated directly.

```javascript
function getAttributeNode(name) {
	// Implementation details...
}
```

---

### Open File with Default Application (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Use the `shell.openPath()` method to open a specified file or folder using the system's default application. This function requires user consent and returns a Promise. Ensure the file extension or an empty string for folders is listed in `manifest.json`'s `launchProcess.extensions`.

```javascript
const { shell } = require("uxp");

// Open a PDF file in the default PDF reader 📄
async function openPDFFile() {
	try {
		// For macOS
		const result = await shell.openPath(
			"/Users/user/Desktop/report.pdf", // 👈 update with your path
			"Opening project report for review",
		);
		// For Windows, use: "C:\\Users\\user\\Desktop\\report.pdf"

		if (result === "") {
			console.log("✅ File opened successfully");
		} else {
			console.error(`❌ Failed to open file: ${result}`);
		}
	} catch (err) {
		console.error("Error opening file:", err);
	}
}

// Open a folder in Finder/Explorer 📁
async function openProjectFolder() {
	// 👇 Must have an empty string "" to allow opening folders
	// in the Manifest requiredPermissions.launchProcess.extensions array
	try {
		// For macOS
		const result = await shell.openPath("/Users/user/Documents/Projects", "Opening project folder");
		// For Windows, use: "C:\\Users\\user\\Documents\\Projects"

		if (result === "") {
			console.log("✅ Folder opened successfully");
		} else {
			console.error(`❌ Failed to open folder: ${result}`);
		}
	} catch (err) {
		console.error("Error opening folder:", err);
	}
}
```

---

### UXP Entrypoints with Promise Support

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/entrypoints

Illustrates the use of Promises in UXP entrypoint hooks (`create`, `show`, `hide`, `destroy`) for asynchronous operations. Supported hooks include `plugin.destroy()`, and `panel.create()`, `panel.show()`, `panel.hide()`, `panel.destroy()`. The `show()` hook is tied to `create()`, and `hide()` to `destroy()`, each with a 300ms timeout.

```javascript
const { entrypoints } = require("uxp");
entrypoints.setup({
	plugin: {
		create() {
			console.log("Plugin create hook");
		},
		destroy() {
			return new Promise(function (resolve, reject) {
				console.log("Plugin destroy hook");
				resolve();
			});
		},
	},
	panels: {
		firstPanel: {
			create(rootNode) {
				return new Promise(function (resolve, reject) {
					console.log("Panel create hook", rootNode);
					resolve();
				});
			},
			show(rootNode, data) {
				return new Promise(function (resolve, reject) {
					console.log("Panel show hook", data);
					resolve();
				});
			},
			hide(rootNode, data) {
				return new Promise(function (resolve, reject) {
					console.log("Panel hide hook", data);
					resolve();
				});
			},
			destroy(rootNode) {
				return new Promise(function (resolve, reject) {
					console.log("Panel destroy hook", rootNode);
					resolve();
				});
			},
		},
	},
	commands: {
		firstCommand: commandHandler,
	},
});
```

---

### Access Plugin Sandbox Data Folder using UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Shows how to access and list contents of a plugin's own data folder using `localFileSystem`. This operation is suitable for storing plugin-specific data and requires the `uxp` module. The `localFileSystem` permission defaults to 'plugin' if not specified.

```javascript
const { localFileSystem } = require("uxp").storage;

async function accessPluginDataFolder() {
	// Access the plugin's data folder
	try {
		const dataFolder = await localFileSystem.getEntryWithUrl("plugin-data:/");
		console.log(`Data folder path: ${dataFolder.nativePath}`);

		// List all files in the data folder
		const entries = await dataFolder.getEntries();
		console.log(`Found ${entries.length} items in data folder`);

		for (const entry of entries) {
			console.log(`- ${entry.name} (${entry.isFile ? "file" : "folder"})`);
		}
	} catch (e) {
		console.error("Failed to access data folder:", e);
	}
}
```

---

### Get Clipboard Content (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/Clipboard

Retrieves data from the clipboard using the getContent method. This is a non-standard API and returns a Promise.

```javascript
navigator.clipboard.getContent();
```

---

### Get Element Attribute Node (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLabelElement

Retrieves the attribute node with a given name. An attribute node represents an attribute in the DOM as a node object.

```javascript
/**
 * Returns the attribute node with the given name.
 *
 * @param {string} name - The name of the attribute node to retrieve.
 * @returns {*} The attribute node, or null if not found.
 */
function getAttributeNode(name) {
	// Implementation details for getting attribute node
}
```

---

### Detect Platform for URL Schemes (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Shows how to check the operating system using `require('os').platform()` to determine the correct URL scheme for platform-specific applications. This ensures that schemes like 'maps://' for macOS and 'bingmaps:' for other platforms are used appropriately.

```javascript
const isMac = require("os").platform() === "darwin";
const scheme = isMac ? "maps://" : "bingmaps:";
```

---

### Get Element Attribute (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLTextAreaElement

Retrieves the value of a specified attribute from an element. Returns null if the attribute is not found. This method is compatible with the standard DOM API.

```javascript
/**
 * Gets the value of the specified attribute.
 * @param {string} name - The name of the attribute whose value to get.
 * @returns {string|null} The attribute's value, or null if it is not set.
 */
Element.prototype.getAttribute(name);
```

---

### Get Element Attribute Node

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLStyleElement

Retrieves the `Attr` object (attribute node) associated with a given attribute name. This provides more detailed information about the attribute than `getAttribute()`.

```javascript
const element = document.getElementById("myElement");
const idAttributeNode = element.getAttributeNode("id");
if (idAttributeNode) {
	console.log("Attribute name:", idAttributeNode.name);
	console.log("Attribute value:", idAttributeNode.value);
}
```

---

### Basic HTML Structure with Spectrum Web Components (SWC)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/user-interfaces

Illustrates the basic HTML structure for utilizing Spectrum Web Components (SWC) after they have been imported. This approach is recommended for production plugins requiring the full Spectrum component set.

```html
<sp-button variant="primary">I'm a SWC button</sp-button> <sp-textfield placeholder="Enter your name"></sp-textfield>
```

---

### Get All Element Attribute Names (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLabelElement

Retrieves an array containing the names of all attributes present on the element. This is useful for iterating over or inspecting an element's attributes.

```javascript
/**
 * Returns the attribute names of the element as an Array of strings.
 *
 * @returns {string[]} An array of attribute names.
 */
function getAttributeNames() {
	// Implementation details for getting attribute names
}
```

---

### Remember User File Selections with UXP Persistent Tokens (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This snippet demonstrates how to use UXP's localFileSystem API to allow users to select a file and then create a persistent token to remember that file across sessions. It also shows how to retrieve the token from localStorage and access the file using the token. This is useful for saving user preferences or frequently accessed files.

```javascript
const { localFileSystem, domains, fileTypes } = require("uxp").storage;

async function selectAndRememberFile() {
	try {
		// Let the user select a file
		const file = await localFileSystem.getFileForOpening({
			initialDomain: domains.userDesktop,
			types: fileTypes.text,
		});

		if (!file) {
			console.log("User cancelled file selection");
			return;
		}

		// Create a persistent token for this file
		const token = await localFileSystem.createPersistentToken(file);

		// Store the token for future use (e.g., in localStorage)
		localStorage.setItem("selectedFileToken", token);

		console.log(`File selected and token saved: ${file.nativePath}`);
	} catch (err) {
		console.error("Failed to create token:", err);
	}
}

async function readPreviouslySelectedFile() {
	try {
		// Retrieve the token from storage
		const token = localStorage.getItem("selectedFileToken");

		if (!token) {
			console.log("No previously selected file found");
			return;
		}

		// Access the file using the token
		const file = await localFileSystem.getEntryForPersistentToken(token);

		// Read the file content
		const content = await file.read();
		console.log(`File content:\n${content}`);
	} catch (err) {
		console.error("Failed to read file using token:", err);
		// Token may be invalid if file was deleted or moved
		localStorage.removeItem("selectedFileToken");
	}
}
```

---

### Configure Multiple Host Applications for UXP Plugin (Development)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/distribution/package

This JSON configuration allows a UXP plugin to be loaded in multiple host applications during development. The UXP Developer Tool packages for the first host in production.

```json
"host": [
  { "app": "premierepro", "minVersion": "25.6.0" },
  { "app": "ps", "minVersion": "25.0.0" }
]
```

---

### Declare localFileSystem Permission in manifest.json

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This snippet demonstrates how to declare the `localFileSystem` permission in your plugin's `manifest.json` file. This is crucial for enabling file system access. The `plugin` value restricts access to the sandbox, `request` allows user permission prompts, and `fullAccess` grants unrestricted access.

```json
{
	// ...
	"requiredPermissions": {
		"localFileSystem": "plugin"
	}
	// ...
}
```

---

### Get Element Attribute by Name

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLStyleElement

Retrieves the value of a specified attribute from an element. Returns a string representing the attribute's value, or an empty string if the attribute is not set.

```javascript
const element = document.getElementById("myDiv");
const className = element.getAttribute("class");
console.log(className);
```

---

### Get the Root Node of an Element using JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLInputElement

Retrieves the root node of the current node, which is the node at the top of the DOM tree. This can be the document itself or a shadow root.

```javascript
const root = element.getRootNode();
```

---

### Get Elements by Class Name (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLabelElement

Returns a live `NodeList` containing all child elements that have the specified class name. This is useful for selecting multiple elements by their class.

```javascript
/**
 * Returns a NodeList of all elements in the document with the specified class name.
 *
 * @param {string} name - The class name to search for.
 * @returns {NodeList} A live NodeList of matching elements.
 */
function getElementsByClassName(name) {
	// Implementation details for getting elements by class name
}
```

---

### Manipulating Element Attributes in JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLElement

Provides methods to get, set, remove, and check for the existence of attributes on an HTML element. These are fundamental for dynamic content modification.

```javascript
const element = document.getElementById("myElement");

// Get an attribute
const className = element.getAttribute("class");

// Set an attribute
element.setAttribute("data-custom", "someValue");

// Remove an attribute
element.removeAttribute("id");

// Check if an attribute exists
const hasDataAttribute = element.hasAttribute("data-custom");

// Get all attribute names
const attributeNames = element.getAttributeNames();

// Get an attribute node
const idAttributeNode = element.getAttributeNode("id");

// Set an attribute node
const newAttribute = document.createAttribute("new-attr");
newAttribute.value = "newValue";
element.setAttributeNode(newAttribute);

// Remove an attribute node
const oldAttributeNode = element.getAttributeNode("data-custom");
if (oldAttributeNode) {
	element.removeAttributeNode(oldAttributeNode);
}
```

---

### Use SWC Button in HTML

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/swc

Render an SWC button element in your HTML markup. This example shows how to create a primary button with custom text.

```html
<sp-button variant="primary">I'm a button</sp-button>
```

---

### Apply CSS to SVG with Variables

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Provides the CSS to accompany the SVG example, defining the '--iconColor' variable to control the fill color of the SVG rectangle.

```css
:root {
	--iconColor: blue;
}
```

---

### Enable Inter-Plugin Communication

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Allows a plugin to communicate with other plugins installed on Adobe Premiere Pro. Set 'enablePluginCommunication' to true to activate this feature.

```json
{
	"enablePluginCommunication": true
}
```

---

### Configure Launch Process Permissions

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Specifies the schemes (e.g., http, https) and file extensions (e.g., pdf) that a plugin is allowed to launch. This is crucial for security and controlling external process execution.

```json
{
	"schemes": ["http", "https"],
	"extensions": ["pdf"]
}
```

---

### Get All Element Attribute Names (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLScriptElement

Retrieves all attribute names present on an element and returns them as an array of strings. This method is useful for iterating over or inspecting all attributes of an element.

```javascript
function getAttributeNames() {
	// Implementation details...
}
```

---

### Get Elements by Tag Name

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLStyleElement

Returns a live `NodeList` containing all descendant elements with the specified tag name. Useful for selecting all elements of a certain type, like all `<div>`s.

```javascript
const paragraphs = document.getElementsByTagName("p");
for (let i = 0; i < paragraphs.length; i++) {
	paragraphs[i].style.fontSize = "16px";
}
```

---

### Get Elements by Class Name/Tag Name (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLTextAreaElement

Methods to retrieve a collection of descendant elements that match a specified class name or tag name. Returns a live NodeList.

```javascript
/**
 * Returns a NodeList of all descendant elements with the specified class name.
 * @param {string} name - The class name to search for.
 * @returns {NodeList} A live NodeList of matching elements.
 */
Element.prototype.getElementsByClassName(name);
```

```javascript
/**
 * Returns a NodeList of all descendant elements with the specified tag name.
 * @param {string} name - The tag name to search for.
 * @returns {NodeList} A live NodeList of matching elements.
 */
Element.prototype.getElementsByTagName(name);
```

---

### Get File for Opening (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider

Shows how to use `getFileForOpening` to retrieve a read-only file from the user's file system. It covers selecting a single file and reading its content, as well as selecting multiple files with specified types.

```javascript
const folder = await fs.getFolder({ initialDomain: domains.userDocuments });
const file = await fs.getFileForOpening({ initialLocation: folder });
if (!file) {
	// no file selected
	return;
}
const text = await file.read();
```

```javascript
const files = await fs.getFileForOpening({ allowMultiple: true, types: fileTypes.images });
if (files.length === 0) {
	// no files selected
}
```

---

### Request Constructor

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/Request

The Request constructor initializes a new Request object, defining the resource to fetch and optional settings.

````APIDOC
## Request Constructor

### Description
Initializes a new `Request` object.

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Parameters Table
| Param | Type | Description |
|---|---|---|
| input | `string` | `Request` | Defines the resource that you wish to fetch. This can either be: A string containing the URL of the resource you want to fetch. A Request object. |
| options | `Object` | '(Optional)' object containing any custom settings that you want to apply to the request. |

### Options Details
- **options.method** (`string`) - Request method. The default is "GET".
- **options.headers** (`string` | `Headers`) - Any headers you want to add to your request.
- **options.body** (`Blob` | `ArrayBuffer` | `TypedArray` | `FormData` | `string` | `ReadableStream` | `URLSearchParams`) - Any body that you want add to your request.
- **options.credentials** (`string`) - Request credentials you want to use for the request. Either "omit" or "include".
- **options.signal** (`AbortSignal`) - AbortSignal object which can be used to abort a request.

### Throws
- `TypeError` If options.body exists and options.method is either "GET" or "HEAD".

### Request Example
```javascript
// Example usage is not provided in the source text.
````

### Response

#### Success Response (200)

Returns a new Request object.

#### Response Example

```json
{
	"example": "new Request object"
}
```

````

--------------------------------

### Get Folder for Access (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider

Demonstrates how to use `getFolder` to prompt the user to select a folder, which can then be used to access its contents (files and subfolders). Files within the selected folder are read-write.

```javascript
const folder = await fs.getFolder();
const myNovel = (await folder.getEntries()).filter(entry => entry.name.indexOf('novel') > 0);
const text = await myNovel.read();
````

---

### Handle Pointer Capture for Element Interaction (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20DOM/Element

Demonstrates how to use setPointerCapture, releasePointerCapture, and hasPointerCapture to manage pointer events for element interaction. This is useful for implementing drag-and-drop or other continuous interaction behaviors. The example includes HTML for a draggable element and JavaScript for handling pointer events.

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

### ReadableStream Constructor

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Streams/ReadableStream

Creates a ReadableStream object with specified underlying source and queuing strategy.

```APIDOC
## ReadableStream(underlyingSource
```

---

### Get/Set Attribute Node (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLTextAreaElement

Methods for getting and setting attribute nodes on an element. `getAttributeNode` returns the attribute node with the specified name, while `setAttributeNode` adds or replaces an attribute node.

```javascript
/**
 * Gets the attribute node with the specified name.
 * @param {string} name - The name of the attribute node to get.
 * @returns {Attr} The attribute node, or null if not found.
 */
Element.prototype.getAttributeNode(name);
```

```javascript
/**
 * Adds or replaces an attribute node.
 * @param {Attr} newAttr - The attribute node to add or replace.
 * @returns {Attr|null} The old attribute node if replaced, otherwise null.
 */
Element.prototype.setAttributeNode(newAttr);
```

```javascript
/**
 * Removes the specified attribute node.
 * @param {Attr} oldAttr - The attribute node to remove.
 * @returns {Attr|null} The removed attribute node, or null if not found.
 */
Element.prototype.removeAttributeNode(oldAttr);
```

---

### Get XMPMeta Object using XMPFile

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPFile

Demonstrates how to use the XMPFile class to obtain an XMPMeta object from a file. This involves creating an XMPFile instance and then calling the getXMP() method. It requires importing the XMPFile class and specifying file path, format, and open flags.

```javascript
const { XMPFile } = require("uxp").xmp;

// Create a new XMPFile object
const xmpFile = new XMPFile("sample.psd", XMPConst.FILE_PHOTOSHOP, XMPConst.OPEN_FOR_UPDATE);

// Get the XMPMeta object
const xmpMeta = xmpFile.getXMP();
```

---

### ProjectSettings - getScratchDiskSettings

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/projectsettings

Retrieves the current scratch disk settings for a project.

```APIDOC
## GET /websites/developer_adobe_premiere-pro_uxp/ProjectSettings/getScratchDiskSettings

### Description
Returns project ScratchDiskSettings.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/ProjectSettings/getScratchDiskSettings

### Parameters
#### Query Parameters
- **project** (_Project_) - Required - The project for which to retrieve scratch disk settings.

### Request Example
GET /websites/developer_adobe_premiere-pro_uxp/ProjectSettings/getScratchDiskSettings?project=project_object

### Response
#### Success Response (200)
- **scratchDiskSettings** (_ScratchDiskSettings_) - The current scratch disk settings for the project.

#### Response Example
{
  "scratchDiskSettings": "scratch_disk_settings_object"
}
```

---

### Copying an Entry to a Folder (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/Entry

Provides examples of using the `copyTo` method to copy an entry (file or folder) to a specified destination folder. It shows basic copying and copying with overwrite enabled.

```javascript
await someFile.copyTo(someFolder);
```

```javascript
await someFile.copyTo(someFolder, { overwrite: true });
```

```javascript
await someFolder.copyTo(anotherFolder, { overwrite: true, allowFolderCopy: true });
```

---

### Import and Use Spectrum Web Components (SWC) in JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/user-interfaces

Demonstrates how to import and use Spectrum Web Components (SWC) in your JavaScript code after installation. This enables the use of Adobe's Spectrum design system components within your UXP plugin.

```javascript
import "@spectrum-web-components/button/sp-button.js";
import "@spectrum-web-components/textfield/sp-textfield.js";
```

---

### Apply Solid Left Border using CSS

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Styles/border-left-style

This example demonstrates how to apply a solid left border to an element using CSS. It sets the width, style, and color of the left border. This is a standard CSS technique applicable in various web development contexts.

```css
.button {
	border-left-width: 2px;
	border-left-style: solid;
	border-left-color: white;
}
```

---

### Get Plugin Folder (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider

Demonstrates how to access the plugin's own folder using `getPluginFolder()`. This folder contains packaged assets and is read-only, intended for accessing bundled resources.

```javascript
const pluginFolder = await fs.getPluginFolder();
```

---

### Implement Platform-Specific Logic for Opening URLs with UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/host-info

This code snippet shows how to implement platform-specific logic for opening external URLs, specifically for map applications. It uses `os.platform()` to detect if the user is on macOS ('darwin') or Windows ('win32') and constructs an appropriate URL scheme. It utilizes the `shell.openExternal` API from UXP.

```javascript
const { host, shell } = require("uxp");
const os = require("os");

// Use platform-appropriate URL scheme 🗺️
async function openMapsLocation(address) {
	const IS_MAC = os.platform() === "darwin";
	let url;
	if (IS_MAC) {
		// Use Apple Maps on macOS
		url = `maps://?address=${encodeURIComponent(address)}`;
	} else {
		// Use Bing Maps on Windows
		url = `bingmaps:?q=${encodeURIComponent(address)}`;
	}

	try {
		await shell.openExternal(url, "Opening maps application");
		console.log(`✅ Opened maps for: ${address}`);
	} catch (err) {
		console.error("Failed to open maps:", err);
	}
}

// Example usage
openMapsLocation("345 Park Ave, San Jose");
```

---

### Register Command Entrypoint in manifest.json

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-commands

This JSON snippet shows how to register a new command entrypoint within the `manifest.json` file. It defines the entrypoint type, a unique ID, and a user-visible label for the command in the Premiere Pro UI.

```json
{
	// ...
	"entrypoints": [
		{
			"type": "command",
			"id": "myCommand",
			"label": "This is a Command"
		}
	]
}
```

---

### Set background-color with CSS (UXP)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Styles/background-color

Example of how to set the background color for an element using CSS within Adobe Premiere Pro UXP. This utilizes standard CSS color formats.

```css
.someElement {
	backgorund-color: blue;
}
```

---

### Modify Existing XMP Metadata and Serialize Compactly

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/getting-started

This example shows how to initialize an XMPMeta object with an existing XMP packet string, update the 'ModifyDate' property with the current date and time, and then serialize the updated packet back into a compact XML string.

```javascript
const { XMPMeta, XMPConst, XMPDateTime } = require("uxp").xmp;
let xmp = new XMPMeta(xmpStr); // Object initialized with xmp packet as string
let dateTime = new XMPDateTime(new Date()); // Now
let oldModificationDate = mp.getProperty(XMPConst.NS_XMP, "ModifyDate", "xmpdate");
console.log("Old modification date: " + oldModificationDate);
xmp.setProperty(XMPConst.NS_XMP, "ModifyDate", dateTime, "xmpdate");

// Serialize to XML, in compact style
let xmpStr = xmp.serialize(XMPConst.SERIALIZE_USE_COMPACT_FORMAT);
```

---

### Attribute Manipulation in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLProgressElement

Allows for getting, setting, removing, and checking the presence of attributes on an element. getAttributeNames returns an array of attribute names. getAttributeNode and setAttributeNode work with attribute nodes.

```javascript
const attributeValue = element.getAttribute(name);
element.setAttribute(name, value);
element.removeAttribute(name);
const hasAttr = element.hasAttribute(name);
const allAttrNames = element.getAttributeNames();
const attrNode = element.getAttributeNode(name);
element.setAttributeNode(newAttr);
element.removeAttributeNode(oldAttr);
```

---

### Initialize Network Request with XMLHttpRequest

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

Initializes or re-initializes an XMLHttpRequest object for making network requests. It supports GET, POST, PUT, and DELETE methods for HTTP/S URLs. Note that UXP requires asynchronous requests, so the 'async' parameter must be true.

```javascript
const xhr = new XMLHttpRequest();
xhr.onload = () => {
	console.log(xhr.response);
	console.log(xhr.responseText);
};
xhr.open("GET", "https://www.myxmlserver.com");
xhr.overrideMimeType("text/plain");
xhr.send();
```

---

### Get Element Attribute (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLScriptElement

Retrieves the value of a specified attribute from an element. It takes the attribute name as a string and returns the attribute's value as a string. If the attribute does not exist, it returns null.

```javascript
function getAttribute(name) {
	// Implementation details...
}
```

---

### Get Element Attribute with JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLVideoElement

Retrieves the value of a specified attribute from an HTML element. Returns the attribute's value as a string, or an empty string if the attribute is not set.

```javascript
element.getAttribute(name);
```

---

### Batch Process Image Metadata with XMPFile API (JavaScript/UXP)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/getting-started

This script iterates through image files in a user-selected folder, reads their XMP metadata, deletes existing creator information, adds a new creator, and writes the modified metadata back to the files. It utilizes the XMPFile, XMPMeta, and XMPConst objects from the 'uxp' library. Ensure you have the necessary UXP development environment set up.

```javascript
console.log( "XMPFiles batch processing example" );
// Load the XMPScript library
const {XMPMeta, XMPConst, XMPFile} = require("uxp").xmp;


// Iterate through the photos in the folder
const uxpfs = require("uxp").storage;
const ufs = uxpfs.localFileSystem;
let folder = await ufs.getFolder({initialDomain: uxpfs.domains.userDocuments});
let files = await folder.getEntries();
files.forEach((file) => {
     console.log( "Process file: " + file.name );


     // Applies only to files, not to folders
     if ( file instanceof Entry ) {
         let xmpFile = new XMPFile( file.nativePath, XMPConst.UNKNOWN, XMPConst.OPEN_FOR_UPDATE );
         let xmp = xmpFile.getXMP();


         // Delete existing authors and add a new one
         // Existing metadata stays untouched
         xmp.deleteProperty( XMPConst.NS_DC, "creator" );
         xmp.appendArrayItem( XMPConst.NS_DC, "creator", "Judy", 0, XMPConst.ARRAY_IS_ORDERED );


         // Write updated metadata into the file
         if ( xmpFile.canPutXMP( xmp ) ) {
             xmpFile.putXMP( xmp );
         }
         xmpFile.closeFile( XMPConst.CLOSE_UPDATE_SAFELY );
     }
}
```

---

### Retrieve Folder Entries (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/Folder

Shows how to get an array of all entries (files and subfolders) contained within a given Folder instance. It includes filtering to identify only files.

```javascript
const entries = await aFolder.getEntries();
const allFiles = entries.filter((entry) => entry.isFile);
```

---

### VideoClipTrackItem Instance Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/videocliptrackitem

Provides documentation for various instance methods of the VideoClipTrackItem class, allowing developers to interact with and modify track items within Adobe Premiere Pro sequences.

````APIDOC
## createAddVideoTransitionAction

### Description
Create add transition action for sequence.

### Method
POST (Implied - Action Creation)

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **videoTransition** (_VideoTransition_) - Required - Description: None
- **addTransitionOptionsProperties** (_AddTransitionOptions_) - Required - Description: None

### Request Example
```json
{
  "videoTransition": "<VideoTransitionObject>",
  "addTransitionOptionsProperties": "<AddTransitionOptionsObject>"
}
````

### Response

#### Success Response (200)

- **Action** (_Action_) - Description: An action object representing the addition of a video transition.

#### Response Example

```json
{
	"action": "<ActionObject>"
}
```

````

```APIDOC
## createMoveAction

### Description
Returns an action that moves the inPoint of the track item to a new time, by shifting it by a number of seconds.

### Method
POST (Implied - Action Creation)

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **tickTime** (_TickTime_) - Required - Description: The target time to move the inPoint to.

### Request Example
```json
{
  "tickTime": "<TickTimeObject>"
}
````

### Response

#### Success Response (200)

- **Action** (_Action_) - Description: An action object representing the movement of the track item's inPoint.

#### Response Example

```json
{
	"action": "<ActionObject>"
}
```

````

```APIDOC
## createRemoveVideoTransitionAction

### Description
Returns true if trackItem has transition.

### Method
POST (Implied - Action Creation)

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **transitionPosition** (_Constants.TransitionPosition_) - Required - Description: Start or end position of transition.

### Request Example
```json
{
  "transitionPosition": "start"
}
````

### Response

#### Success Response (200)

- **Action** (_Action_) - Description: An action object representing the removal of a video transition.

#### Response Example

```json
{
	"action": "<ActionObject>"
}
```

````

```APIDOC
## createSetDisabledAction

### Description
Returns an action that enables/disables the trackItem.

### Method
POST (Implied - Action Creation)

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **disabled** (_boolean_) - Required - Description: Boolean value to set the disabled state.

### Request Example
```json
{
  "disabled": true
}
````

### Response

#### Success Response (200)

- **Action** (_Action_) - Description: An action object representing the enable/disable state change of the track item.

#### Response Example

```json
{
	"action": "<ActionObject>"
}
```

````

```APIDOC
## createSetEndAction

### Description
Create set end time action for sequence.

### Method
POST (Implied - Action Creation)

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **tickTime** (_TickTime_) - Required - Description: The target end time.

### Request Example
```json
{
  "tickTime": "<TickTimeObject>"
}
````

### Response

#### Success Response (200)

- **Action** (_Action_) - Description: An action object representing the setting of the track item's end time.

#### Response Example

```json
{
	"action": "<ActionObject>"
}
```

````

```APIDOC
## createSetInPointAction

### Description
Create SetInPointAction for setting the track item in point relative to the start time of the project item referenced by this track item.

### Method
POST (Implied - Action Creation)

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **tickTime** (_TickTime_) - Required - Description: The target in-point time relative to the project item start.

### Request Example
```json
{
  "tickTime": "<TickTimeObject>"
}
````

### Response

#### Success Response (200)

- **Action** (_Action_) - Description: An action object representing the setting of the track item's in-point.

#### Response Example

```json
{
	"action": "<ActionObject>"
}
```

````

```APIDOC
## createSetNameAction

### Description
Returns an action that renames the trackItem.

### Method
POST (Implied - Action Creation)

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **inName** (_string_) - Required - Description: The new name for the track item.

### Request Example
```json
{
  "inName": "New Track Item Name"
}
````

### Response

#### Success Response (200)

- **Action** (_Action_) - Description: An action object representing the renaming of the track item.

#### Response Example

```json
{
	"action": "<ActionObject>"
}
```

````

```APIDOC
## createSetOutPointAction

### Description
Create SetOutPointAction for setting the track item out point relative to the start time of the project item referenced by this track item.

### Method
POST (Implied - Action Creation)

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **tickTime** (_TickTime_) - Required - Description: The target out-point time relative to the project item start.

### Request Example
```json
{
  "tickTime": "<TickTimeObject>"
}
````

### Response

#### Success Response (200)

- **Action** (_Action_) - Description: An action object representing the setting of the track item's out-point.

#### Response Example

```json
{
	"action": "<ActionObject>"
}
```

````

```APIDOC
## createSetStartAction

### Description
Create set start time action for sequence.

### Method
POST (Implied - Action Creation)

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **tickTime** (_TickTime_) - Required - Description: The target start time.

### Request Example
```json
{
  "tickTime": "<TickTimeObject>"
}
````

### Response

#### Success Response (200)

- **Action** (_Action_) - Description: An action object representing the setting of the track item's start time.

#### Response Example

```json
{
	"action": "<ActionObject>"
}
```

````

```APIDOC
## getComponentChain

### Description
Returns VideoComponentChain.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/componentChain

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **VideoComponentChain** (_VideoComponentChain_) - Description: The VideoComponentChain associated with the track item.

#### Response Example

```json
{
	"videoComponentChain": "<VideoComponentChainObject>"
}
```

````

```APIDOC
## getDuration

### Description
Returns timecode representing the duration of this track item relative to the sequence start.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/duration

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **TickTime** (_TickTime_) - Description: The duration of the track item.

#### Response Example

```json
{
	"duration": "<TickTimeObject>"
}
```

````

```APIDOC
## getEndTime

### Description
Returns a TickTime object representing the ending sequence time of this track item relative to the sequence start time.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/endTime

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **TickTime** (_TickTime_) - Description: The end time of the track item.

#### Response Example

```json
{
	"endTime": "<TickTimeObject>"
}
```

````

```APIDOC
## getInPoint

### Description
Returns a TickTime object representing the track item in point relative to the start time of the project item referenced by this track item.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/inPoint

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **TickTime** (_TickTime_) - Description: The in-point of the track item.

#### Response Example

```json
{
	"inPoint": "<TickTimeObject>"
}
```

````

```APIDOC
## getIsSelected

### Description
Returns if trackItem is selected or not.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/isSelected

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **boolean** (_boolean_) - Description: True if the track item is selected, false otherwise.

#### Response Example

```json
{
	"isSelected": true
}
```

````

```APIDOC
## getMatchName

### Description
Returns the value of internal matchname for this trackItem.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/matchName

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **string** (_string_) - Description: The internal match name of the track item.

#### Response Example

```json
{
	"matchName": "InternalTrackItemName"
}
```

````

```APIDOC
## getMediaType

### Description
Returns UUID representing the underlying media type of this track item.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/mediaType

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **Guid** (_Guid_) - Description: The UUID representing the media type.

#### Response Example

```json
{
	"mediaType": "<UUID_String>"
}
```

````

```APIDOC
## getName

### Description
Returns the display name for trackItem.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/name

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **string** (_string_) - Description: The display name of the track item.

#### Response Example

```json
{
	"name": "Track Item Name"
}
```

````

```APIDOC
## getOutPoint

### Description
Returns a TickTime object representing the track item out point relative to the start time of the project item referenced by this track item.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/outPoint

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **TickTime** (_TickTime_) - Description: The out-point of the track item.

#### Response Example

```json
{
	"outPoint": "<TickTimeObject>"
}
```

````

```APIDOC
## getProjectItem

### Description
Returns the project item for this track item.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/projectItem

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **ProjectItem** (_ProjectItem_) - Description: The ProjectItem object associated with the track item.

#### Response Example

```json
{
	"projectItem": "<ProjectItemObject>"
}
```

````

```APIDOC
## getSpeed

### Description
Returns the value of speed of the trackItem.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/speed

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **number** (_number_) - Description: The speed value of the track item.

#### Response Example

```json
{
	"speed": 1.0
}
```

````

```APIDOC
## getStartTime

### Description
Returns a TickTime object representing the starting sequence time of this track item relative to the sequence start time.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/startTime

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **TickTime** (_TickTime_) - Description: The start time of the track item.

#### Response Example

```json
{
	"startTime": "<TickTimeObject>"
}
```

````

```APIDOC
## getTrackIndex

### Description
Index representing the track index of the track this track item belongs to.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/trackIndex

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **number** (_number_) - Description: The index of the track.

#### Response Example

```json
{
	"trackIndex": 1
}
```

````

```APIDOC
## getType

### Description
Index representing the type of this track item.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/type

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **number** (_number_) - Description: The type index of the track item.

#### Response Example

```json
{
	"type": 0
}
```

````

```APIDOC
## isAdjustmentLayer

### Description
Returns true if the trackitem is an adjustment layer.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/isAdjustmentLayer

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **boolean** (_boolean_) - Description: True if the track item is an adjustment layer, false otherwise.

#### Response Example

```json
{
	"isAdjustmentLayer": false
}
```

````

```APIDOC
## isDisabled

### Description
Returns true if trackitem is muted/disabled.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/isDisabled

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **boolean** (_boolean_) - Description: True if the track item is disabled, false otherwise.

#### Response Example

```json
{
	"isDisabled": false
}
```

````

```APIDOC
## isSpeedReversed

### Description
Returns true if the trackitem is reversed.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/VideoClipTrackItem/{trackItemId}/isSpeedReversed

### Parameters
#### Path Parameters
- **trackItemId** (string) - Required - Description: The ID of the track item.

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
null
````

### Response

#### Success Response (200)

- **number** (_number_) - Description: True if the speed is reversed, false otherwise.

#### Response Example

```json
{
	"isSpeedReversed": false
}
```

````

--------------------------------

### Get Elements by Tag Name (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLabelElement

Returns a live `NodeList` containing all child elements with the specified tag name. This is useful for selecting elements of a particular type, like 'div' or 'span'.

```javascript
/**
 * Returns a NodeList of all elements in the document with the specified tag name.
 *
 * @param {string} name - The tag name to search for (e.g., 'div', 'span').
 * @returns {NodeList} A live NodeList of matching elements.
 */
function getElementsByTagName(name) {
  // Implementation details for getting elements by tag name
}
````

---

### Get Elements by Class Name (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLScriptElement

Returns a live `NodeList` containing all descendant elements of the parent element that have the specified class name. The class name is a string, and multiple class names can be specified, separated by spaces.

```javascript
function getElementsByClassName(name) {
	// Implementation details...
}
```

---

### Get XMPPacketInfo Object using XMPFile

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPFile

Illustrates retrieving XMPPacketInfo from a file using the XMPFile class. This includes creating an XMPFile object and calling the getPacketInfo() method. The output logs the length, offset, padding size, and character form of the XMP packet.

```javascript
const { XMPFile } = require("uxp").xmp;

// Create a new XMPFile object
const xmpFile = new XMPFile("sample.psd", XMPConst.FILE_PHOTOSHOP, XMPConst.OPEN_FOR_UPDATE);

// Get XMPPacketInfo object
const xmpPacketInfo = xmpFile.getPacketInfo();
console.log(xmpPacketInfo.length);
console.log(xmpPacketInfo.offset);
console.log(xmpPacketInfo.padSize);
console.log(xmpPacketInfo.charForm);
```

---

### Get Temporary Folder Instance (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/Folder

Demonstrates how to obtain a Folder instance representing the temporary directory using the local file system provider. This is useful for temporary file storage during plugin execution.

```javascript
const fs = require("uxp").storage.localFileSystem;
const folder = await fs.getTemporaryFolder(); // Gets the Folder instance
console.log(folder.isFolder); // returns true
```

---

### Component Instance Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/component

This section details the instance methods available for a Component object, allowing developers to retrieve information about the component and its parameters.

```APIDOC
## Component Instance Methods

### getDisplayName

Retrieves the display name for the component.

### Method
GET

### Endpoint
`/websites/developer_adobe_premiere-pro_uxp/component/{id}/displayName` (Hypothetical endpoint, actual endpoint may vary)

### Response
#### Success Response (200)
- **displayName** (string) - The display name of the component.

### getMatchName

Retrieves the internal match name for the component.

### Method
GET

### Endpoint
`/websites/developer_adobe_premiere-pro_uxp/component/{id}/matchName` (Hypothetical endpoint, actual endpoint may vary)

### Response
#### Success Response (200)
- **matchName** (string) - The internal match name of the component.

### getParam

Retrieves a parameter from the component based on the provided index.

### Method
GET

### Endpoint
`/websites/developer_adobe_premiere-pro_uxp/component/{id}/param/{paramIndex}` (Hypothetical endpoint, actual endpoint may vary)

#### Parameters
##### Path Parameters
- **paramIndex** (number) - Required - The zero-based index of the parameter to retrieve.

### Response
#### Success Response (200)
- **param** (ComponentParam) - The requested component parameter.

### getParamCount

Gets the total number of parameters in the component.

### Method
GET

### Endpoint
`/websites/developer_adobe_premiere-pro_uxp/component/{id}/paramCount` (Hypothetical endpoint, actual endpoint may vary)

### Response
#### Success Response (200)
- **paramCount** (number) - The number of parameters in the component.

```

---

### Monitor XMLHttpRequest Ready State Changes

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

The readyState property indicates the current state of the XMLHttpRequest object during a request. States range from UNSENT (0) to DONE (4). This example logs the readyState each time it changes.

```javascript
const xhr = new XMLHttpRequest();
xhr.onreadystatechange = () => {
	console.log(xhr.readyState);
};
xhr.open("GET", "https://www.adobe.com");
xhr.send();
```

---

### Get All Response Headers with XMLHttpRequest

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

Retrieves all response headers as a single string. Headers are sorted and combined, with values separated by ', '. Each header is formatted as '[lowercase-name]: [value]\r\n'. This method is typically called when readyState is XMLHttpRequest.HEADERS_RECEIVED.

```javascript
const xhr = new XMLHttpRequest();
xhr.onreadystatechange = () => {
	if (xhr.readyState === XMLHttpRequest.HEADERS_RECEIVED) {
		console.log(xhr.getAllResponseHeaders());
	}
};
xhr.open("GET", "https://www.adobe.com");
xhr.send();
```

---

### Basic Text Field with Label - UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-textfield

Renders a basic text field with an optional associated label. This example shows how to create a text field with a placeholder and a required label using the `<sp-textfield>` and `<sp-label>` components.

```html
<sp-textfield placeholder="Phone Number">
	<sp-label
		isrequired="true"
		slot="label"
		>Phone Number</sp-label
	>
</sp-textfield>
```

---

### UXP Plugin Manifest Permissions for Local File System Access (JSON)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This snippet shows the necessary configuration within a UXP plugin's manifest.json file to grant the plugin access to the local file system. Specifically, it highlights the 'requiredPermissions' section, setting 'localFileSystem' to 'plugin' to enable file operations within the plugin's sandbox and data directories.

```json
{
	"manifestVersion": 5,
	// ...
	"requiredPermissions": {
		"localFileSystem": "plugin"
	}
	// ...
}
```

---

### Get a Specific Entry from a Folder (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/Folder

Explains how to retrieve a specific file or folder entry from within a Folder using its name or path. This is useful for accessing existing items.

```javascript
const myNovel = await aFolder.getEntry("mynovel.txt");
```

---

### Sequence Instance Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/sequence

Perform actions on a sequence, including clearing selections, creating clones, setting in/out points, and managing tracks.

````APIDOC
## Sequence Instance Methods

### clearSelection

### Description
Clears the selection of track items within the sequence.

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
- Returns `true` if selection was cleared successfully, `false` otherwise.

#### Response Example
```json
true
````

````

```APIDOC
## Sequence Instance Methods

### createCloneAction

### Description
Creates an action to clone the given sequence.

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
- **Action** - An object representing the action to clone the sequence.

#### Response Example
```json
{
  "type": "CloneAction"
}
````

````

```APIDOC
## Sequence Instance Methods

### createSetInPointAction

### Description
Creates an action to set the in-point for the sequence.

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
- **tickTime** (_TickTime_) - The time at which to set the in-point.

### Request Example
```json
{
  "tickTime": {
    "seconds": 10,
    "ticks": 0
  }
}
````
