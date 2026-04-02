### Write and Read Files to Desktop using Node.js fs (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This JavaScript code snippet shows how to export data to the user's Desktop and read data from a file on the Desktop using the Node.js 'fs' module. It includes error handling and OS-specific path examples. Requires 'localFileSystem' with 'fullAccess' in the manifest.

```javascript
const fs = require("fs");

async function exportToDesktop() {
	// Export data to the user's Desktop
	try {
		const exportData = "Exported data from plugin";

		// For macOS
		await fs.writeFile(
			"/Users/user/Desktop/export.txt", // 👇 update with your user
			exportData,
			{ encoding: "utf-8" },
		);
		// For Windows, use: "C:/Users/user/Desktop/export.txt"

		console.log("File exported to Desktop");
	} catch (e) {
		console.error("Failed to export file:", e);
	}
}

async function readFromDesktop() {
	// Read a file from the user's Desktop folder
	try {
		// For macOS
		const content = await fs.readFile(
			"/Users/user/Desktop/export.txt", // 👇 update with your user
			"utf8",
		);
		// For Windows, use: "C:/Users/user/Desktop/export.txt"

		console.log("File content:", content);
	} catch (e) {
		console.error("Failed to read file:", e);
	}
}
```

---

### Install, Remove, List UXP Plugins via UPIA on Windows

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/distribution/install

This snippet shows how to manage UXP plugins on Windows using the UnifiedPluginInstallerAgent.exe command-line tool. It includes commands for installing, removing, and listing plugins, requiring administrator privileges and correct file paths.

```batch
cd "C:\Program Files\Common Files\Adobe\Adobe Desktop Common\RemoteComponents\UPI\UnifiedPluginInstallerAgent"

UnifiedPluginInstallerAgent.exe /help
UnifiedPluginInstallerAgent.exe /version
UnifiedPluginInstallerAgent.exe /install <extension-file-path>
UnifiedPluginInstallerAgent.exe /remove <extension-file-path>
UnifiedPluginInstallerAgent.exe /list <all || product display name>

# Examples:
UnifiedPluginInstallerAgent.exe /install "C:\Temp\Test-xjluvc_premierepro.ccx"
UnifiedPluginInstallerAgent.exe /remove "startup-test"
UnifiedPluginInstallerAgent.exe /list all
```

---

### Installing and Importing Spectrum Web Component Button

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum

Shows the necessary steps to use Adobe Spectrum Web Components (SWC) in UXP. This involves installing the component via npm and then importing it before use.

```bash
npm i @swc-uxp-wrappers/button
```

```javascript
import "@swc-uxp-wrappers/button/sp-button.js";
```

---

### Install, Remove, List UXP Plugins via UPIA on macOS

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/distribution/install

This snippet demonstrates how to use the UnifiedPluginInstallerAgent on macOS to manage UXP plugins. It covers installing, removing, and listing plugins using command-line arguments. Ensure you have the correct paths to the extension files and necessary admin privileges.

```shell
cd "/Library/Application Support/Adobe/Adobe Desktop Common/RemoteComponents/UPI/UnifiedPluginInstallerAgent/UnifiedPluginInstallerAgent.app/Contents/macOS"

./UnifiedPluginInstallerAgent --help
./UnifiedPluginInstallerAgent --version
./UnifiedPluginInstallerAgent --install <extension-file-path>
./UnifiedPluginInstallerAgent --remove <extension-file-path>
./UnifiedPluginInstallerAgent --list <all || product display name>

# Examples:
./UnifiedPluginInstallerAgent --install "~/Desktop/Test-xjluvc_premierepro.ccx"
./UnifiedPluginInstallerAgent --remove "startup-test"
./UnifiedPluginInstallerAgent --list all
```

---

### Working with Premiere Application Objects

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference

This section details how to access and interact with core Premiere Pro objects like the application, project, and sequence using the UXP API. It provides examples for getting the active project and active sequence.

````APIDOC
## Working with Premiere Objects

### Premiere Application

The `app` object provides access to the rest of Premiere's objects and methods.

#### Getting the Active Project

To get the currently active project:

```javascript
const project = await app.Project.getActiveProject();
````

#### Getting the Active Sequence

From the project object, you can get the active sequence:

```javascript
const sequence = await project.getActiveSequence();
```

````

--------------------------------

### Sliding Element Example (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLSelectElement

Example demonstrating how to implement a sliding element using pointer events and pointer capture. It includes functions for starting, stopping, and handling pointer movements to update the element's position.

```javascript
// HTML
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


// JS
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

---

### Initialize UXP Entrypoints Setup

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/entrypoints

This JavaScript code demonstrates the basic initialization of the UXP `entrypoints.setup()` method, which is crucial for handling plugin and panel invocations. It imports the `entrypoints` object from the 'uxp' module.

```javascript
const { entrypoints } = require("uxp");
entrypoints.setup({
	/* ... */
});
```

---

### Example UXP Plugin with Multiple Modules

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/js-modules

This example demonstrates a UXP plugin's main entry point (`index.js`) importing and utilizing functions from separate module files (`video-processor.js` and `ui-helpers.js`). It showcases asynchronous operations and error handling within commands.

```javascript
const { entrypoints } = require("uxp");
const { processVideo } = require("./lib/video-processor.js");
const { showNotification } = require("./lib/ui-helpers.js");

entrypoints.setup({
	commands: {
		processCommand: async () => {
			try {
				await processVideo();
				showNotification("Video processing completed!");
			} catch (error) {
				showNotification("Error: " + error.message);
			}
		},
	},
});
```

```javascript
async function processVideo() {
	// Video processing logic here
	console.log("Processing video...");
	// Simulate async operation
	return new Promise((resolve) => setTimeout(resolve, 1000));
}

function getVideoInfo() {
	// Return video information
	return { duration: 120, format: "mp4" };
}

module.exports = {
	processVideo,
	getVideoInfo,
};
```

```javascript
function showNotification(message) {
	console.log(`Notification: ${message}`);
	// Additional UI notification logic
}

function createProgressBar() {
	// Progress bar creation logic
	return document.createElement("progress");
}

module.exports = {
	showNotification,
	createProgressBar,
};
```

---

### Access User Documents Folder with Full Access using UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Demonstrates accessing arbitrary file system locations, such as the user's Documents folder, using UXP's `localFileSystem` with the `fullAccess` permission. This requires specifying absolute paths and setting `"localFileSystem": "fullAccess"` in `manifest.json`. The code provides examples for macOS and Windows.

```javascript
const { localFileSystem } = require("uxp").storage;

async function accessUserDocuments() {
	// Access a specific location outside the sandbox
	try {
		// For macOS
		const documentsFolder = await localFileSystem.getEntryWithUrl(
			"file:/Users/user/Documents", // 👈 update with your user
		);
		// For Windows, use: "file:/C:/Users/user/Documents"

		console.log(`Documents folder path: ${documentsFolder.nativePath}`);

		// Read a specific file
		const configFile = await localFileSystem.getEntryWithUrl(
			"file:/Users/user/Documents/config.json", // 👈 update with your user
		);
		if (configFile.isFile) {
			const content = await configFile.read();
			console.log("Config file content:", content);
		}
	} catch (e) {
		console.error("Failed to access documents folder:", e);
	}
}
```

---

### Install Spectrum Web Components (SWC) using npm

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/user-interfaces

Instructions for installing specific Spectrum Web Components (SWC) for use in UXP plugins. It's crucial to lock components to version 0.37.0 for compatibility with the current UXP version.

```bash
npm install @spectrum-web-components/button@0.37.0
npm install @spectrum-web-components/textfield@0.37.0
```

---

### createSetStartAction API

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/media

Creates an action to set the start time of the media.

````APIDOC
## POST /websites/developer_adobe_premiere-pro_uxp/media/actions/setStart

### Description
Returns an action that sets the start time of the media. This method is available from version 25.0 onwards.

### Method
POST

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/media/actions/setStart

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **time** (_TickTime_) - The time to set as the start of the media.

### Request Example
```json
{
  "time": "00:00:02:00"
}
````

### Response

#### Success Response (200)

- **action** (_Action_) - An action object that can be executed to set the media start time.

#### Response Example

```json
{
	"action": "setStartAction"
}
```

````

--------------------------------

### Install TypeScript and Create Source Folder

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/typescript-support

Installs the TypeScript compiler as a development dependency and creates a 'src' directory for your TypeScript source files.

```bash
npm install --save-dev typescript
mkdir src
````

---

### Get Entry with URL using LocalFileSystem API

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This JavaScript code snippet shows how to obtain an `Entry` reference for a file using its URL with the `localFileSystem.getEntryWithUrl()` method. This is a common pattern when working with the `LocalFileSystem` API, which uses object references for file operations.

```javascript
const dataFile = await localFileSystem.getEntryWithUrl("plugin-data:/settings.json");
```

---

### HTML Canvas Setup and JavaScript Interaction

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLCanvasElement

This snippet shows how to set up an HTML canvas element and use JavaScript to retrieve its height, width, and 2D rendering context. It also includes an example of drawing a triangle using the 2D context. This code is intended for use within a UXP environment.

```html
<sp-body id="layers">
	<canvas
		id="canvas"
		height="230"
		width="1500"
	></canvas>
</sp-body>
<footer>
	<sp-button
		id="btnPopulateLoadScript"
		onclick="show_height()"
		>Canvas Height</sp-button
	>
	<sp-button
		id="btnPopulateLoadScript"
		onclick="show_width()"
		>Canvas Width</sp-button
	>
	<sp-button
		id="btnPopulateLoadScript"
		onclick="getContext()"
		>Get Context</sp-button
	>
</footer>
```

```javascript
const canvas = document.getElementById("canvas");

function show_height() {
	console.log("Canvas Height: " + canvas.height);
}

function show_width() {
	console.log("Canvas Width: " + canvas.width);
}

// Function to get the canvas context and draw a triangle using lines
function getContext() {
	let context = canvas.getContext("2d"); // get's the canvas context

	// Draw a triangle. For more details on the below apis refer to interfaces such as CanvasRenderingContext2D, CanvasGradient. The details of the interfaces are shared as a link at the top of this documentation
	context.beginPath();
	context.moveTo(50, 50);
	context.lineTo(100, 50);
	context.lineTo(100, 100);
	context.lineTo(50, 50);
	context.closePath();
	context.stroke();
}
```

---

### Install SWC Button Component using npm

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/swc

Install the Spectrum Web Component for a button into your project using npm. This command fetches the necessary package from the npm registry.

```bash
npm install @swc-uxp-wrappers/button
```

---

### fetch() Network Permissions Limitation Example

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/fetch

This example demonstrates the limitation in UXP v7.4.0 onwards regarding the use of wildcards in top-level domains within the `permissions.network.domains` configuration. It shows incorrect usage with wildcards in TLDs.

```json
"domains": [ "https://www.adobe.*", "https://www.*" ]
```

---

### Access LocalFileSystem and Types API

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This JavaScript snippet shows how to import the `localFileSystem` and `types` objects from the `uxp` module. These are the primary components for interacting with the file system using the object-oriented `LocalFileSystem` API.

```javascript
const { localFileSystem, types } = require("uxp").storage;
```

---

### UXP Plugin and Panel Setup in JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Entry%20Points/EntryPoints

This snippet demonstrates the basic structure for setting up a UXP plugin and its panels. It includes defining plugin lifecycle methods (create, destroy) and panel-specific methods (create, show, hide, destroy, invokeMenu, update, validateNode). It also shows how to define menu items for panels.

```javascript
const { entrypoints } = require("uxp");

entrypoints.setup({
    plugin: {
        create() { /* ... */ },
        destroy() { /* ... */ }
    },
    panels: {
        "panel1": {
            create() { /* ... */ },
            show() { /* ... */ },
            hide() { /* ... */ },
            destroy() { /* ... */ },
            invokeMenu() { /* ... */ },
            update() { /* ... */ }, // customEntrypoint example
            validateNode() { /* ... */ } // customEntrypoint example
            menuItems: [
                {
                    id: "signIn",
                    label: "Sign In...",
                    enabled: false,
                    checked: false,
                    submenu: [
                        { id: "submenu1", label: "submenu1", enabled: false, checked: false },
                        { "submenu2" }
                    ]
                },
                "-",  // separator.
                "Sign out",  // by default enabled, and the id will be same with the label.
            ]
        },
        "panel2": { /* ... */ }
    },
    commands: {
        "command1": {
            run() { /* ... */ },
            cancel() { /* ... */ }
        },
        "command2": function() { /* ... */ }
    }
});
```

---

### Element Pointer Capture and Event Handling Example (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLDialogElement

Demonstrates how to use setPointerCapture, releasePointerCapture, addEventListener, and removeEventListener to implement a draggable element. This example utilizes pointer events to manage the drag-and-drop functionality.

```javascript
// HTML structure is assumed to be present in the document.
// Example: <div id="slider">SLIDE ME</div>

function beginSliding(e) {
	slider.setPointerCapture(e.pointerId);
	slider.addEventListener("pointermove", slide);
}

function stopSliding(e) {
	slider.releasePointerCapture(e.pointerId);
	slider.removeEventListener("pointermove", slide);
}

function slide(e) {
	slider.style.left = e.clientX + "px"; // Ensure unit is added
}

const slider = document.getElementById("slider");

slider.addEventListener("pointerdown", beginSliding);
slider.addEventListener("pointerup", stopSliding);
```

---

### Send GET Request (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

Demonstrates how to send a GET request to a server and log the response text. It utilizes the XMLHttpRequest object to handle the request and load event.

```javascript
const xhr = new XMLHttpRequest();
xhr.addEventListener("load", () => {
	console.log(xhr.responseText);
});
xhr.open("GET", "https://www.adobe.com");
xhr.send();
```

---

### Shell Module - Best Practices

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Guidelines for using the shell module, including providing clear context, handling user denial, checking platform compatibility, encoding URL parameters, and declaring necessary schemes and extensions.

````APIDOC
## Best Practices

### 1. Provide Clear Context

When using the `developerText` parameter in the `shell.openExternal` method, ensure the explanation is user-friendly and in plain language.

```javascript
// ✅ Good: Clear and specific
await shell.openExternal(
  "https://example.com/guide",
  "Opening tutorial guide in your browser"
);

// ❌ Avoid: Vague or technical
await shell.openExternal(
  "https://example.com/guide",
  "Opening URL"
);
````

### 2. Handle User Denial Gracefully

Users can deny launch requests. Always check the return value of shell methods and provide alternative options if the operation fails or is denied.

```javascript
const result = await shell.openPath(filePath, "Opening project file");
if (result !== "") {
	// User denied or operation failed
	console.log("Unable to open file. Please open it manually.");
}
```

### 3. Check Platform Compatibility

Utilize platform detection to ensure the correct URL schemes are used for different operating systems.

```javascript
const isMac = require("os").platform() === "darwin";
const scheme = isMac ? "maps://" : "bingmaps:";
```

### 4. Encode URL Parameters

When constructing URLs with query parameters, always encode special characters to prevent issues.

```javascript
const subject = encodeURIComponent("My Subject");
const url = `mailto:user@example.com?subject=${subject}`;
```

### 5. Declare All Schemes and Extensions

In your `manifest.json`, only declare the schemes and file extensions that your plugin actively uses to avoid requesting unnecessary permissions.

````

--------------------------------

### Use URL Schemes for File System Access

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Illustrates the use of UXP's URL schemes to access specific file system locations. `plugin:/`, `plugin-data:/`, and `plugin-temp:/` provide shortcuts to sandbox directories with varying read/write permissions. `file:/` allows access to arbitrary locations but requires `fullAccess` permission.

```html
<img src="plugin:/icons/logo.png" />
<img src="file:/Users/user/Downloads/sample.png" />
<!-- update the path based on your system -->
````

---

### Working with XMP Date and Structs using XMPMeta

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPMeta

Illustrates creating an XMPMeta object and using struct-based APIs, including setting and managing XMPDateTime objects and structure fields. This example requires 'uxp'.

```javascript
let { XMPDateTime, XMPMeta, XMPConst } = require("uxp").xmp;
let meta = new XMPMeta();

let jsDate = new Date("2007-04-10T17:54:50+01:00");
let xmpDate = new XMPDateTime(jsDate);
meta.setProperty(XMPConst.NS_XMP, "CreateDate", xmpDate, XMPConst.XMPDATE);
meta.doesPropertyExist(XMPConst.NS_XMP, "CreateDate");
let prop = meta.getProperty(XMPConst.NS_XMP, "CreateDate", XMPConst.XMPDATE);
meta.deleteProperty(XMPConst.NS_XMP, "CreateDate");

meta.setStructField(XMPConst.NS_XML, "structNameSample", XMPConst.NS_XMP, "sampleFieldName", "sampleFieldValue");
if (meta.doesStructFieldExist(XMPConst.NS_XML, "structNameSample", XMPConst.NS_XMP, "sampleFieldName")) {
	prop = meta.getStructField(XMPConst.NS_XML, "structNameSample", XMPConst.NS_XMP, "sampleFieldName");
	meta.deleteStructField(XMPConst.NS_XML, "structNameSample", XMPConst.NS_XMP, "sampleFieldName");
	if (meta.doesStructFieldExist(XMPConst.NS_XML, "structNameSample", XMPConst.NS_XMP, "sampleFieldName")) {
		console.log("Struct field exists");
	} else {
		console.log("Struct field doesn't exist");
	}
} else {
	console.log("Struct field doesn't exist");
}
```

---

### Shell Module - Summary

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

An overview of the UXP Shell module, including security considerations for launching processes, the two primary shell methods (`openPath` and `openExternal`), and user experience best practices.

```APIDOC
## Summary

### 1. Launch Process Security

By default, plugins cannot launch external processes. You must declare the `launchProcess` permission in `manifest.json`.

*   **`extensions`**: An array of file extensions your plugin can open (e.g., `[".pdf", ".mp4"]`). Use `""` to allow opening folders.
*   **`schemes`**: An array of URL schemes your plugin can use (e.g., `["https", "mailto"]`).
*   User consent is **always required** for security. Users will see a dialog for each launch attempt unless the choice is remembered.

### 2. Two Shell Methods

*   **`shell.openPath(path, developerText)`**: Opens files or folders in their default system application. Requires the file extension to be listed in `extensions`.
*   **`shell.openExternal(url, developerText)`**: Launches applications via URL schemes. Requires the scheme to be listed in `schemes`. Cannot use `file://` (use `openPath()` instead).

### 3. User Experience Best Practices

*   URL schemes are operating system-specific; use `require("os").platform()` to check the OS and select the appropriate scheme.
*   Provide clear, user-friendly text in the `developerText` parameter.
*   Handle user denial gracefully by checking return values and offering fallback options.
*   Only declare schemes and extensions your plugin actually uses.
*   Encode URL parameters with `encodeURIComponent()` to handle special characters properly.
```

---

### Open User-Selected File with UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Presents a file picker dialog to the user, allowing them to select a text file to open. It then reads and logs the content of the selected file and its native path. Requires 'request' level permission for localFileSystem.

```javascript
const { localFileSystem, domains, fileTypes } = require("uxp").storage;

async function openUserSelectedFile() {
	// Present a file picker starting at the user's Desktop
	try {
		const file = await localFileSystem.getFileForOpening({
			initialDomain: domains.userDesktop,
			types: fileTypes.text,
		});

		if (!file) {
			console.log("User cancelled file selection");
			return;
		}

		// Read the selected file
		const content = await file.read();
		console.log(`File content:\n${content}`);
		console.log(`File path: ${file.nativePath}`);
	} catch (err) {
		console.error("Failed to open file:", err);
	}
}
```

---

### Get XMLHttpRequest Response URL

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

The responseURL property returns the URL of the server's response after any redirects. It is an empty string if the URL is null. This example shows how to log the response URL after a GET request.

```javascript
const xhr = new XMLHttpRequest();
xhr.onload = () => {
	console.log(xhr.responseURL);
};
xhr.open("GET", "https://www.adobe.com");
xhr.send();
```

---

### Pointer Interaction Example: Sliding Element

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHtmlElement

Demonstrates how to use `setPointerCapture` and `releasePointerCapture` along with pointer events to implement a draggable element. This example includes HTML structure and JavaScript event handling for pointer down, move, and up.

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
	slider.style.left = e.clientX + "px"; // Added 'px' for correct CSS
}

const slider = document.getElementById("slider");

slider.addEventListener("pointerdown", beginSliding);
slider.addEventListener("pointerup", stopSliding);

// Added pointercancel event for robustness
slider.addEventListener("pointercancel", stopSliding);
```

---

### Get Plugin User ID using UXP JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/User%20Information

This JavaScript code demonstrates how to retrieve the unique GUID of the plugin user using the `userId()` method from the `uxp.userInfo` API. The output is a string representing the user's GUID.

```javascript
let userId = require("uxp").userInfo.userId(); // Get the GUID of plugin user
console.log(userId); // e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

---

### Get Elements by Tag Name (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLScriptElement

Returns a live `NodeList` containing all descendant elements of the parent element that have the specified tag name. The tag name is a string, for example, 'div', 'p', or 'span'.

```javascript
function getElementsByTagName(name) {
	// Implementation details...
}
```

---

### Premiere Pro UXP API Overview

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference

This section provides an overview of how to access the Premiere Pro DOM via UXP. It includes an example of requiring the Premiere Pro module and explains the general capabilities of the API.

````APIDOC
## Overview

The Premiere Pro UXP API allows you to interact with the Premiere Pro DOM. You can open documents, modify them, run menu items, and more.

### Getting Started

To access the Premiere Pro DOM, you need to require the `premierepro` module:

```javascript
const app = require('premierepro');
````

### Key Features

- Access to Premiere Pro objects and methods.
- Ability to manipulate projects and sequences.
- Understanding of synchronous and asynchronous operations.

````

--------------------------------

### Play Media with JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLVideoElement

Initiates media playback for a video element. It returns a resolved Promise upon starting playback and emits a 'play' event. An error event is emitted if playback cannot start.

```javascript
let vid = document.getElementById("sampleVideo");
vid.play();
vid.addEventListener("play", (ev) => {
    console.log("Event - play");
});
````

---

### Pointer Capture Example (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLinkElement

Demonstrates how to use setPointerCapture and releasePointerCapture to manage pointer events for dragging or interactive elements. Includes HTML and CSS for a draggable element.

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
/**
 * Sets pointer capture for the element. This implementation does not dispatch the `gotpointercapture` event on the element.
 * @param {number} pointerId - The unique identifier of the pointer to be captured.
 * @throws {DOMException} If the element is not connected to the DOM.
 */
function setPointerCapture(pointerId) {
	// Implementation details...
}

/**
 * Releases pointer capture for the element. This implementation does not dispatch the `lostpointercapture` event on the element.
 * @param {number} pointerId - The unique identifier of the pointer to be released.
 */
function releasePointerCapture(pointerId) {
	// Implementation details...
}

// Example Usage:
function beginSliding(e) {
	slider.setPointerCapture(e.pointerId);
	slider.addEventListener("pointermove", slide);
}

function stopSliding(e) {
	slider.releasePointerCapture(e.pointerId);
	slider.removeEventListener("pointermove", slide);
}

function slide(e) {
	slider.style.left = e.clientX + "px"; // Ensure units are added
}

const slider = document.getElementById("slider");

slider.addEventListener("pointerdown", beginSliding);
slider.addEventListener("pointerup", stopSliding);
```

---

### HTML h3 Element Example

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-html/Hierarchy/h3

Demonstrates the basic usage of the HTML h3 element in UXP. This element is not theme-aware and requires manual styling for theme adaptation.

```html
<h3>Hello, world!</h3>
```

---

### Get XMLHttpRequest Status Text

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

The statusText property returns the string associated with the HTTP status code (e.g., 'OK', 'Not Found'). It is an empty string if the request is in the UNSENT or OPENED state. This example logs the status text after a successful response.

```javascript
const xhr = new XMLHttpRequest();
xhr.onload = () => {
	console.log(xhr.statusText);
};
xhr.open("GET", "https://www.adobe.com");
xhr.send();
```

---

### Implement Spectrum Web Component Button

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Shows how to use a Spectrum Web Component, specifically a button, within a UXP plugin. Requires manual installation, import, and bundling of the component.

```html
<sp-button variant="primary"> Click me </sp-button>
```

---

### Implement Plugin and Panel Lifecycle Hooks in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-lifecycle-hooks

This JavaScript code demonstrates how to implement lifecycle hooks for both the plugin and its panels using the `entrypoints.setup()` method in UXP. It includes examples for plugin creation and destruction, and panel creation, showing, hiding, and destruction. Promises are supported for asynchronous operations. The `rootNode` parameter is available for panel hooks.

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
			// 👈 matches the panel ID from manifest.json
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
});
```

---

### Implement UXP Plugin Lifecycle Hooks

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/entrypoints

This JavaScript code shows how to define 'create' and 'destroy' lifecycle hooks for a UXP plugin using `entrypoints.setup()`. These functions are called when the plugin container is created and destroyed, respectively, allowing for setup and teardown logic.

```javascript
const { entrypoints } = require("uxp");
entrypoints.setup({
	plugin: {
		create() {
			console.log("Plugin created");
		},
		destroy() {
			console.log("Plugin destroyed");
		},
	},
});
```

---

### Instance Methods for Footage Interpretation

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/footageinterpretation

This section covers the instance methods available for getting and setting various footage interpretation properties.

```APIDOC
## Instance Methods

### getAlphaUsage

#### Description
Get alpha usage type property of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The alpha usage type.

### getFieldType

#### Description
Get field type of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The field type.

### getFrameRate

#### Description
Get frame rate of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The frame rate.

### getIgnoreAlpha

#### Description
Get ignore alpha property of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_boolean_) - Whether alpha is ignored.

### getInputLUTID

#### Description
Get input LUTID of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_string_) - The input LUTID.

### getInvertAlpha

#### Description
Get invert alpha property of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_boolean_) - Whether alpha is inverted.

### getPixelAspectRatio

#### Description
Get pixel aspect ratio of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The pixel aspect ratio.

### getRemovePullDown

#### Description
Get removePullDown property of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_boolean_) - Whether pull-down is removed.

### getVrConform

#### Description
Get vr conform projection type of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The VR conform projection type.

### getVrHorzView

#### Description
Get VR horizontal view of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The VR horizontal view value.

### getVrLayout

#### Description
Get VR layout type of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The VR layout type.

### getVrVertView

#### Description
Get VR vertical view of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The VR vertical view value.

### setAlphaUsage

#### Description
Set alpha usage type property of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **alphaUsage** (_number_) - The alpha usage type to set.

### setFieldType

#### Description
Set field type of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **fieldType** (_number_) - The field type to set.

### setFrameRate

#### Description
Set frame rate of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **frameRate** (_number_) - The frame rate to set.

### setIgnoreAlpha

#### Description
Set ignore alpha property of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **ignoreAlpha** (_boolean_) - Whether to ignore alpha.

### setInputLUTID

#### Description
Set input LUTID of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **inputLUTID** (_string_) - The input LUTID to set.

### setInvertAlpha

#### Description
Set invert alpha property of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **invertAlpha** (_boolean_) - Whether to invert alpha.

### setPixelAspectRatio

#### Description
Set pixel aspect ratio of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **pixelAspectRatio** (_number_) - The pixel aspect ratio to set.

### setRemovePullDown

#### Description
Set removePullDown property of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **removePulldown** (_boolean_) - Whether to remove pull-down.

### setVrConform

#### Description
Set VR conform projection type of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **vrConform** (_number_) - The VR conform projection type to set.

### setVrHorzView

#### Description
Set VR horizontal view of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **vrHorzView** (_number_) - The VR horizontal view value to set.

### setVrLayout

#### Description
Set VR layout type of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **vrLayOut** (_number_) - The VR layout type to set.

### setVrVertView

#### Description
Set VR vertical view of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **vrVertView** (_number_) - The VR vertical view value to set.
```

---

### Static Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/project

Methods for project creation, retrieval, and opening.

```APIDOC
## Static Methods

### createProject

#### Description
Create a new project.

#### Parameters
- **path** (string) - The path to create the new project.

### getActiveProject

#### Description
Returns the currently active project.

### getProject

#### Description
Get project referenced by given UID.

#### Parameters
- **projectGuid** (Guid) - The unique identifier of the project to retrieve.

### open

#### Description
Open a project.

#### Parameters
- **path** (string) - The path to the project file.
- **openProjectOptions** (OpenProjectOptions) - Options for opening the project.
```

---

### Combining Spectrum UXP Widgets and SWC in Adobe UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/faqs

This example demonstrates how to seamlessly integrate Spectrum UXP Widgets and Spectrum Web Components (SWC) within an Adobe UXP environment. It showcases the usage of both `<sp-banner>` (SWC) and `<sp-dropdown>` (Spectrum UXP Widget) in the same HTML structure.

```html
<sp-banner>
	<!-- Spectrum Web Components -->
	<div slot="header">Header text</div>
	<div slot="content">Content of the banner</div>
</sp-banner>
<sp-dropdown
	placeholder="Select an option"
	style="width: 320px"
>
	<!-- Spectrum UXP Widget -->
	<sp-menu slot="options">
		<sp-menu-item> Option 1 </sp-menu-item>
		<sp-menu-item> Option 2 </sp-menu-item>
	</sp-menu>
</sp-dropdown>
```

---

### Import UXP XMP Module

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/getting-started

This snippet demonstrates how to import the necessary XMP module from the 'uxp' library. This is the first step required to utilize any of the XMP scripting API functionalities.

```javascript
const xmp = require("uxp").xmp;
```

---

### Styling with :defined in UXP CSS - HTML and CSS Example

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Pseudo-classes/defined

This example demonstrates how to apply styles using the :defined pseudo-class in UXP. It includes HTML for a custom element and a standard paragraph, along with CSS rules that target both elements and specific ones.

```html
<simple-custom text="Custom element example text"></simple-custom>

<p>Standard paragraph example text</p>
```

```css
/* Give the `p` elements distinctive background */
p {
	background: yellow;
}

/* Both the custom and the built-in element are given italic text */
:defined {
	font-style: italic;
}

/* Only simple-custom element is applied with green background*/
simple-custom:defined {
	display: block;
	background: green;
}
```

---

### Select Folder for Export with UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Allows the user to select a folder for batch export operations. Once a folder is selected, it creates three text files within that folder and writes content to each. Requires 'request' level permission for localFileSystem.

```javascript
const { localFileSystem, domains, fileTypes } = require("uxp").storage;

async function selectFolderForExport() {
	// Let the user select a folder for batch export
	try {
		const folder = await localFileSystem.getFolder({
			initialDomain: domains.userDocuments,
		});

		if (!folder) {
			console.log("User cancelled folder selection");
			return;
		}

		// Create multiple files in the selected folder
		for (let i = 1; i <= 3; i++) {
			const file = await folder.createFile(`export_${i}.txt`, { overwrite: true });
			await file.write(`Content for file ${i}`);
		}

		console.log(`Created 3 files in: ${folder.nativePath}`);
	} catch (err) {
		console.error("Failed to export files:", err);
	}
}
```

---

### WebView Load Start Event Listener (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

Shows how to attach an event listener to a WebView element to detect when the loading process has begun. The 'loadstart' event provides the URL that is being loaded.

```javascript
const webview = document.getElementById("webviewSample");
// Print the url when loading has started
webview.addEventListener("loadstart", (e) => {
	console.log(`webview.loadstart ${e.url}`);
});
```

---

### Instance Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/project

Methods for managing project state and content.

```APIDOC
## Instance Methods

### close

#### Description
Close a project.

#### Parameters
- **closeProjectOptions** (CloseProjectOptions) - Options for closing the project.

### createSequence

#### Description
Create a new sequence with the default preset path.

#### Parameters
- **name** (string) - The name of the new sequence.
- **presetPath** (string) - Deprecated: use createSequenceWithPresetPath() instead.

### createSequenceFromMedia

#### Description
Create a new sequence with a given name and media.

#### Parameters
- **name** (string) - The name of the new sequence.
- **clipProjectItems** (ClipProjectItem[]) - An array of media items to include in the sequence.
- **targetBin** (ProjectItem) - The bin where the new sequence should be created.

### deleteSequence

#### Description
Delete a given sequence from the project.

#### Parameters
- **sequence** (Sequence) - The sequence to delete.

### executeTransaction

#### Description
Execute an undoable transaction by passing a compound action.

#### Parameters
- **callback** ((compoundAction: CompoundAction) => void) - The callback function containing the compound action.
- **undoString** (string)? - An optional string to describe the transaction for undo history.

### getActiveSequence

#### Description
Get the active sequence of the project.

### getColorSettings

#### Description
Get the project color settings object.

### getInsertionBin

#### Description
Get the current insertion bin.

### getRootItem

#### Description
The root item of the project, which contains all items of the project on the lowest level.

### getSequence

#### Description
Get a sequence by its ID from the project.

#### Parameters
- **guid** (Guid) - The unique identifier of the sequence.

### getSequences

#### Description
Get an array of all sequences in this project.

### importAEComps

#### Description
Import After Effects compositions into the project.

#### Parameters
- **aepPath** (string) - The path to the After Effects project file.
- **compNames** (string[]) - An array of composition names to import.
- **TargetBin** (ProjectItem) - The bin to import the compositions into.

### importAllAEComps

#### Description
Import all After Effects compositions from a project file.

#### Parameters
- **aepPath** (string) - The path to the After Effects project file.
- **TargetBin** (ProjectItem) - The bin to import the compositions into.

### importFiles

#### Description
Import files into the root or target bin of the project.

#### Parameters
- **filePaths** (string[]) - An array of file paths to import.
- **suppressUI** (boolean) - Whether to suppress the user interface during import.
- **targetBin** (ProjectItem) - The bin to import the files into.
- **asNumberedStills** (boolean) - Whether to import files as numbered stills.

### importSequences

#### Description
Import sequences from a project file.

#### Parameters
- **projectPath** (string) - The path to the project file containing the sequences.
- **sequenceIds** (Guid[]) - An array of sequence IDs to import.

### lockedAccess

#### Description
Get read/upgrade locked access to the Project. The project state will not change during the execution of the callback function. Can call executeTransaction while having locked access.

#### Parameters
- **callback** (() => void) - The callback function to execute with locked access.

### openSequence

#### Description
Open a sequence and return true if successful.

#### Parameters
- **sequence** (Sequence) - The sequence to open.

### pauseGrowing

#### Description
Pause growing of files instead of swapping the files.

#### Parameters
- **pause** (boolean) - Whether to pause growing.

### save

#### Description
Save the project.

### saveAs

#### Description
Save the project at the provided path.

#### Parameters
- **path** (string) - The path to save the project to.

### setActiveSequence

#### Description
Set the active sequence of the project.

#### Parameters
- **sequence** (Sequence) - The sequence to set as active.
```

---

### Get Entry Metadata using UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/EntryMetadata

Demonstrates how to retrieve metadata for a plugin folder using the `localFileSystem` and `getMetadata()` method in UXP. This involves accessing the local file system, getting a folder instance, and then logging its name from the metadata.

```javascript
const fs = require("uxp").storage.localFileSystem;
const folder = await fs.getPluginFolder(); // Gets an instance of Folder (or Entry)
const entryMetaData = await folder.getMetadata();
console.log(entryMetaData.name);
```

---

### Element Attribute Handling (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLAnchorElement

Provides methods to get, set, remove, and check for the presence of attributes on an element. This includes getting attribute names, specific attribute nodes, and setting attribute nodes.

```javascript
/**
 * Gets the value of a specified attribute on the element.
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
 * Removes a specified attribute from the element.
 * @param {string} name - Name of the attribute to remove.
 */
function removeAttribute(name) {
	// Implementation details would go here
}

/**
 * Checks if the element has a specified attribute.
 * @param {string} name - Name of the attribute to check for.
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
 * @returns {Array<string>} An array of attribute names.
 */
function getAttributeNames() {
	// Implementation details would go here
}

/**
 * Gets the attribute node with the specified name.
 * @param {string} name - The name of the attribute node to get.
 * @returns {*} The attribute node.
 */
function getAttributeNode(name) {
	// Implementation details would go here
}

/**
 * Sets the attribute node for the element.
 * @param {*} newAttr - The attribute node to set.
 */
function setAttributeNode(newAttr) {
	// Implementation details would go here
}

/**
 * Removes the attribute node from the element.
 * @param {*} oldAttr - The attribute node to remove.
 */
function removeAttributeNode(oldAttr) {
	// Implementation details would go here
}
```

---

### VideoTrack Instance Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/videotrack

Details the instance methods available for the VideoTrack, including getting the track index, media type, track items, and mute state.

```APIDOC
## VideoTrack Instance Methods

### getIndex

#### Description
Returns the index of the track within its track group.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Returns
- **number** - The index representing the track's position.

### getMediaType

#### Description
Returns the unique identifier for the underlying media type of the track.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Returns
- **Guid** - UUID representing the media type.

### getTrackItems

#### Description
Retrieves an array of VideoClipTrackItem objects from the track.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Parameters
##### Query Parameters
- **trackItemType** (Constants.TrackItemType) - Required - Specifies the type of track items to retrieve (e.g., Empty, Clip, Transition, Preview, Feedback).
- **includeEmptyTrackItems** (boolean) - Optional - Whether to include empty track items in the results.

#### Returns
- **VideoClipTrackItem[]** - An array of VideoClipTrackItem objects.

### isMuted

#### Description
Checks if the track is currently muted.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Returns
- **boolean** - True if the track is muted, false otherwise.

### setMute

#### Description
Sets the mute state of the track.

#### Method
POST

#### Endpoint
N/A (Instance Method)

#### Parameters
##### Query Parameters
- **mute** (boolean) - Required - Set to true to mute the track, false to unmute.
```

---

### Associate Command Entrypoint with Handler using entrypoints.setup()

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-commands

This code demonstrates how to associate a command entrypoint with its handler function using the `entrypoints.setup()` method in `index.js`. This approach is suitable for plugins with multiple entrypoint types. Ensure `entrypoints.setup()` is called only once.

```javascript
const { entrypoints } = require("uxp");

function myCommandHandler() {
	console.log("Command invoked!");
}

entrypoints.setup({
	commands: {
		myCommand: myCommandHandler,
	},
});
```

---

### Get File Instance using UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/File

Demonstrates how to obtain a File instance using the local file system module in UXP. This involves requiring the 'uxp' module and using 'createEntryWithUrl' to get a File object, which can then be checked for its 'isFile' property.

```javascript
const fs = require("uxp").storage.localFileSystem;
const file = await fs.createEntryWithUrl("file:/Users/user/Documents/tmp"); // Gets a File instance
console.log(file.isFile); // returns true
```

---

### Importing Module-Based UXP Core APIs (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis

Shows how to import module-based UXP Core APIs using require(). This includes importing the main UXP module, the file system module (fs), and the operating system utilities module (os).

```javascript
// Parent UXP module
const uxp = require("uxp");

// File system access
const fs = require("fs");

// Operating system utilities
const os = require("os");
```

---

### UXP Plugin Manifest Example (JSON)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

This JSON structure defines the manifest for a UXP plugin. It includes metadata like version, ID, name, host application compatibility, entry points (commands and panels), icons, and network permissions.

```json
{
	"manifestVersion": 5,
	"id": "YOUR_ID_HERE",
	"name": "Name of your plugin",
	"version": "1.0.0",
	"main": "index.html",
	"host": {
		"app": "HOST_APPLICATION",
		"minVersion": "HOST_VERSION"
	},
	"entrypoints": [
		{
			"type": "command",
			"id": "commandFn",
			"label": {
				"default": "Show A Dialog"
			}
		},
		{
			"type": "panel",
			"id": "panelName",
			"label": {
				"default": "Panel Name"
			}
		}
	],
	"icons": [
		{
			"width": 24,
			"height": 24,
			"path": "icons/icon.png",
			"scale": [1, 2]
		}
	],
	"requiredPermissions": {
		"network": {
			"domains": "all"
		}
	}
}
```

---

### Enable User Info Access in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

This snippet shows how to enable and access anonymized user GUID information in UXP plugins. It requires setting 'enableUserInfo' to true in the manifest and then using 'uxp.userInfo.userId()' to retrieve the GUID.

```javascript
// set "enableUserInfo" to true in the manifest

let userId = require("uxp").userInfo.userId(); // Get the GUID of user
console.log(userId);
// "dad8483a3682a0c3e0fa990281142353901a69fc371254edde8b7dd38ca604c6"
```

---

### Applying Styling with UXP and Premiere APIs (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis

This example outlines the conceptual use of both Premiere APIs for modifying clip properties and UXP APIs for UI feedback, such as showing progress, when applying styles to elements within Premiere Pro.

```javascript
// Premiere APIs to modify the clip properties
// UXP APIs to show progress in your plugin's UI
```

---

### Read and Write Files in Plugin Sandbox with UXP 'fs' Module (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This snippet demonstrates how to use the 'fs' module, which provides a Node.js-like interface for file operations within the UXP plugin environment. It shows how to read a configuration file from the plugin's directory using the 'plugin:/' URL scheme and how to write data to the plugin's data folder using 'plugin-data:/'.

```javascript
const fs = require("fs");

async function readConfigFile() {
	// Read a configuration file from the plugin folder
	try {
		const content = await fs.readFile("plugin:/config.json", "utf8");
		const config = JSON.parse(content);
		console.log("Configuration loaded:", config);
	} catch (e) {
		console.error("Failed to read config file:", e);
	}
}

async function writeToDataFolder() {
	// Write data to the plugin's data folder
	try {
		const data = {
			lastRun: new Date().toISOString(),
			version: "1.0.0",
		};

		await fs.writeFile("plugin-data:/state.json", JSON.stringify(data, null, 2), "utf-8");

		console.log("State saved successfully");
	} catch (e) {
		console.error("Failed to save state:", e);
	}
}
```

---

### Mixing HTML, SWC, and Spectrum UXP Widgets

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum

Provides an example of seamlessly integrating plain HTML elements, Spectrum Web Components (SWC), and built-in Spectrum UXP widgets within a single form structure.

```html
<form>
	<sp-banner>
		<div slot="header">Header text</div>
		<div slot="content">Content of the banner</div>
	</sp-banner>
	<sp-button variant="primary">I'm a button</sp-button>
</form>
```

---

### Configure Local File System Access (JSON)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This JSON snippet configures the necessary permissions for a UXP plugin to access the local file system. It specifies 'localFileSystem' with 'fullAccess', allowing the plugin to read and write to any location on the user's file system.

```json
{
	"manifestVersion": 5,
	// ...
	"requiredPermissions": {
		"localFileSystem": "fullAccess"
	}
	// ...
}
```

---

### Media Properties API

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/media

Access and retrieve properties of the media, including its start time and duration.

````APIDOC
## Media Properties API

### Description
This API provides access to the properties of a media item, such as its start time and duration.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/media

### Parameters
#### Path Parameters
None

#### Query Parameters
None

### Request Example
```json
{}
````

### Response

#### Success Response (200)

- **start** (_TickTime_) - The start time of the media.
- **duration** (_TickTime_) - The duration of the media.

#### Response Example

```json
{
	"start": "00:00:01:00",
	"duration": "00:00:10:00"
}
```

````

--------------------------------

### Reading a File using UXP Core APIs (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis

This snippet demonstrates how to read content from a file using UXP Core APIs. It uses the 'fs' module to get a file object for opening and then reads its content asynchronously.

```javascript
const fs = require("fs");
const file = await fs.getFileForOpening();
const content = await file.read();
````

---

### Create File in Folder using UXP LocalFileSystem

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Demonstrates how to create a new file within a specified folder using UXP's `localFileSystem`. It includes error handling and checks if the entry is indeed a folder before creating the file. Requires the `uxp` module.

```javascript
const { localFileSystem, types } = require("uxp").storage;

async function createFileInFolder() {
	try {
		// Create a folder entry
		const folderEntry = await localFileSystem.createEntryWithUrl("plugin-temp:/myFolder", { type: types.folder });

		// Verify it's a folder before using folder-specific methods
		if (folderEntry.isFolder) {
			const newFile = await folderEntry.createFile("data.txt", { overwrite: true });
			await newFile.write("This is sample content.");
			console.log(`File created at: ${newFile.nativePath}`);
		}
	} catch (e) {
		console.error("Failed to create file:", e);
	}
}
```

---

### Log Host Environment Information with UXP and Node.js

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/host-info

This snippet demonstrates how to log detailed host environment information, including OS, application name and version, UXP runtime version, plugin version, and UI locale. It requires the `uxp` and `os` modules. The output is logged to the console.

```javascript
const { host, versions } = require("uxp");
const os = require("os");

// Log host environment information 💻
function logHostInfo() {
	console.log("=== Host Environment ===");
	console.log(`OS: ${os.platform()} ${os.release()}`);
	console.log(`Application: ${host.name} v${host.version}`);
	console.log(`UXP Runtime: v${versions.uxp}`);
	console.log(`Plugin Version: v${versions.plugin}`);
	console.log(`UI Locale: ${host.uiLocale}`);
}
logHostInfo();
```

---

### Get File/Folder Stats Asynchronously - JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/fs

Gets information asynchronously about a file or folder at a given path. The returned object contains statistics similar to Node.js's `Stats` class, though some properties might be platform-dependent. If no callback is provided, it returns a Promise.

```javascript
const stats = await fs.lstat("plugin-data:/textFile.txt");
const isFile = stats.isFile();
```

---

### getNamespacePrefix - Get Namespace Prefix

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPMeta

Retrieves the prefix associated with a registered namespace URI.

````APIDOC
## getNamespacePrefix(namespaceURI)

### Description
Retrieves the prefix associated with a registered namespace URI.

### Method
`getNamespacePrefix`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **namespaceURI** (string) - Required - The namespace URI string. Refer to Schema namespace string constants.

### Request Example
```javascript
XMPMeta.getNamespacePrefix(namespaceURI)
````

### Response

#### Success Response (200)

- **Return Value** (string) - The prefix string, followed by a colon.

#### Response Example

```json
{
	"return": "prefix:"
}
```

````

--------------------------------

### ProjectSettings - createSetScratchDiskSettingsAction

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/projectsettings

Returns an action which sets ScratchDiskSettings for a project.

```APIDOC
## POST /websites/developer_adobe_premiere-pro_uxp/ProjectSettings/createSetScratchDiskSettingsAction

### Description
Returns an action which sets ScratchDiskSettings for a project.

### Method
POST

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/ProjectSettings/createSetScratchDiskSettingsAction

### Parameters
#### Request Body
- **project** (_Project_) - Required - The project for which to set scratch disk settings.
- **scratchDiskSettings** (_ScratchDiskSettings_) - Required - The scratch disk settings to apply.

### Request Example
{
  "project": "project_object",
  "scratchDiskSettings": "scratch_disk_settings_object"
}

### Response
#### Success Response (200)
- **action** (_Action_) - An action object that can be invoked to set the scratch disk settings.

#### Response Example
{
  "action": "action_object"
}
````

---

### Headers Manipulation Methods

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/Headers

Methods for appending, deleting, getting, and setting header values.

```APIDOC
## append(name, value)

### Description
Appends a new value onto an existing header or adds the header if it does not exist.

### Method
`append`

### Parameters
- **name** (`string`) - Required - The name of the HTTP header.
- **value** (`string`) - Required - The value of the HTTP header.
```

```APIDOC
## delete(name)

### Description
Deletes a header from the current Header object.

### Method
`delete`

### Parameters
- **name** (`string`) - Required - The name of the HTTP header.
```

```APIDOC
## get(name)

### Description
Returns a byte string of all values of a header within the Headers object. Returns null if the header does not exist.

### Method
`get`

### Returns
- `string` - The value of the retrieved header, or null if not found.

### Parameters
- **name** (`string`) - Required - The name of the HTTP header.
```

```APIDOC
## set(name, value)

### Description
Sets a new value for an existing header or adds the header if it does not exist.

### Method
`set`

### Parameters
- **name** (`string`) - Required - The name of the HTTP header.
- **value** (`string`) - Required - The value of the HTTP header.
```

---

### fetch() Method Signature and Parameters

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/fetch

This documentation outlines the signature and parameters for the window.fetch() method. It details the 'input' argument (URL string or Request object) and the optional 'init' object, which allows customization of HTTP request methods, headers, body, and credentials.

```javascript
window.fetch(input, [init])

Parameters:
- input: `string` or `Request` - The URL string or a Request object.
- [init]: `Object` (Optional) - Custom settings for the HTTP request.
  - [init.method]: `string` - HTTP request method (default: "GET").
  - [init.headers]: `Headers` - HTTP request headers.
  - [init.body]: `string` | `ArrayBuffer` | `TypedArray` | `Blob` | `FormData` | `URLSearchParams` - Request body.
  - [init.credentials]: `string` - Indicates whether to send cookies ("omit" or "include").
```

---

### Attribute Manipulation

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHtmlElement

Methods for getting, setting, and checking element attributes.

````APIDOC
## getAttribute(name)

### Description
Returns the value of a specified attribute on the element.

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
// const value = element.getAttribute('data-custom');
````

### Response

#### Success Response (200)

- **string** - The value of the attribute, or null if the attribute is not set.

#### Response Example

```json
"some-value"
```

````

```APIDOC
## setAttribute(name, value)

### Description
Sets the value of a specified attribute on the element. If the attribute already exists, its value is updated; otherwise, a new attribute is added.

### Method
PUT (conceptually, as it modifies state)

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
// element.setAttribute('data-custom', 'new-value');
````

### Response

#### Success Response (200)

None (modifies element state)

#### Response Example

None

````

```APIDOC
## removeAttribute(name)

### Description
Removes a specified attribute from the element.

### Method
DELETE (conceptually, as it removes state)

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
// element.removeAttribute('data-custom');
````

### Response

#### Success Response (200)

None (modifies element state)

#### Response Example

None

````

```APIDOC
## hasAttribute(name)

### Description
Returns a boolean indicating whether the element has the specified attribute.

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
// const hasAttr = element.hasAttribute('data-custom'); // true or false
````

### Response

#### Success Response (200)

- **boolean** - True if the attribute exists, false otherwise.

#### Response Example

```json
true
```

````

```APIDOC
## hasAttributes()

### Description
Returns a boolean value indicating whether the current element has any attributes or not.

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
// const hasAnyAttributes = element.hasAttributes(); // true or false
````

### Response

#### Success Response (200)

- **boolean** - True if the element has any attributes, false otherwise.

#### Response Example

```json
false
```

````

```APIDOC
## getAttributeNames()

### Description
Returns the attribute names of the element as an Array of strings.

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
// const attributeNames = element.getAttributeNames(); // ['id', 'class', ...]
````

### Response

#### Success Response (200)

- **Array** - An array of strings, where each string is an attribute name.

#### Response Example

```json
["id", "class", "data-info"]
```

````

```APIDOC
## getAttributeNode(name)

### Description
Returns the attribute node object for the specified attribute.

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
// const attrNode = element.getAttributeNode('id');
````

### Response

#### Success Response (200)

- **\* ** - The attribute node object, or null if the attribute is not found.

#### Response Example

```json
{
	"name": "id",
	"value": "myElement"
}
```

````

```APIDOC
## setAttributeNode(newAttr)

### Description
Adds or replaces a new attribute node in the element.

### Method
PUT (conceptually, as it modifies state)

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
// const newAttr = document.createAttribute('data-custom');
// newAttr.value = 'example';
// element.setAttributeNode(newAttr);
````

### Response

#### Success Response (200)

- **\* ** - The previous attribute node if it was replaced, otherwise null.

#### Response Example

```json
null
```

````

```APIDOC
## removeAttributeNode(oldAttr)

### Description
Removes an attribute node from the element.

### Method
DELETE (conceptually, as it removes state)

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
// const attrNodeToRemove = element.getAttributeNode('data-custom');
// if (attrNodeToRemove) {
//   element.removeAttributeNode(attrNodeToRemove);
// }
````

### Response

#### Success Response (200)

- **\* ** - The removed attribute node.

#### Response Example

```json
{
	"name": "data-custom",
	"value": "example"
}
```

````

--------------------------------

### Shell Module - Troubleshooting Common Issues

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Solutions for common problems encountered when using the shell module, including permission errors, silent failures, platform-specific scheme issues, and file access restrictions.

```APIDOC
## Troubleshoot Common Issues

| Symptom                       | Likely Cause                      | Solution                                                     |
|-------------------------------|-----------------------------------|--------------------------------------------------------------|
| Permission denied error       | Missing manifest entry            | Add the extension or scheme to `launchProcess` in manifest. |
| Operation fails silently      | User denied consent               | Check return value and handle denial gracefully.             |
| Platform-specific scheme not working | Wrong scheme for OS             | Use platform detection to choose the correct scheme.         |
| `file://` scheme doesn't work | Wrong method used                 | Use `openPath()` for local files, not `openExternal()`.      |
| UWP restrictions on Windows   | System security policy            | UWP apps can only access files in their sandbox.            |
````

---

### Element Attribute Manipulation

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLButtonElement

A set of methods for getting, setting, and removing attributes from an HTML element. Includes methods to retrieve attribute values by name, set or update attributes, remove attributes, check for attribute existence, get all attribute names, and manage attribute nodes.

```javascript
/**
 * Returns the value of the specified attribute on the element.
 * @param {string} name - Name of the attribute whose value you want to get.
 * @returns {string} The value of the attribute.
 */
function getAttribute(name) {
	// Implementation details...
	return ""; // Placeholder
}

/**
 * Sets the value of a specified attribute on the element.
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
 * Returns a boolean value indicating whether the element has the specified attribute.
 * @param {string} name - Name of the attribute to check for.
 * @returns {boolean} True if the attribute exists, false otherwise.
 */
function hasAttribute(name) {
	// Implementation details...
	return false; // Placeholder
}

/**
 * Returns a boolean value indicating whether the current element has any attributes or not.
 * @returns {boolean} True if the element has any attributes, false otherwise.
 */
function hasAttributes() {
	// Implementation details...
	return false; // Placeholder
}

/**
 * Returns an array of strings representing all attribute names of the element.
 * @returns {Array<string>} An array of attribute names.
 */
function getAttributeNames() {
	// Implementation details...
	return []; // Placeholder
}

/**
 * Returns the attribute node with the specified name.
 * @param {string} name - The name of the attribute node to retrieve.
 * @returns {Attr | null} The attribute node or null if not found.
 */
function getAttributeNode(name) {
	// Implementation details...
	return null; // Placeholder
}

/**
 * Sets a new attribute node to the element.
 * @param {Attr} newAttr - The attribute node to set.
 * @returns {Attr | null} The previous attribute node if one existed, otherwise null.
 */
function setAttributeNode(newAttr) {
	// Implementation details...
	return null; // Placeholder
}

/**
 * Removes the specified attribute node from the element.
 * @param {Attr} oldAttr - The attribute node to remove.
 * @returns {Attr | null} The removed attribute node or null if not found.
 */
function removeAttributeNode(oldAttr) {
	// Implementation details...
	return null; // Placeholder
}
```

---

### Interactive Element Dragging Example (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLabelElement

Demonstrates how to implement drag-and-drop functionality for an HTML element using pointer events, setPointerCapture, and releasePointerCapture. This allows users to click and drag an element within the viewport.

```javascript
// HTML
/*
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
*/

// JS
function beginSliding(e) {
	slider.setPointerCapture(e.pointerId);
	slider.addEventListener("pointermove", slide);
}

function stopSliding(e) {
	slider.releasePointerCapture(e.pointerId);
	slider.removeEventListener("pointermove", slide);
}

function slide(e) {
	// Note: In a real scenario, you might want to calculate deltaX/deltaY
	// and adjust based on the initial click position for smoother dragging.
	slider.style.left = e.clientX + "px"; // Using clientX for simplicity
}

const slider = document.getElementById("slider");

slider.addEventListener("pointerdown", beginSliding);
slider.addEventListener("pointerup", stopSliding);

// Adding pointercancel is good practice for robustness
slider.addEventListener("pointercancel", stopSliding);
```

---

### Attribute Manipulation API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLDialogElement

APIs for getting, setting, and checking element attributes.

```APIDOC
## getAttribute(name)

### Description
Returns the value of a specified attribute on the element.

### Method
`getAttribute`

### Parameters
#### Path Parameters
- **name** (string) - The name of the attribute whose value you want to get.

### Returns
`string`

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttribute

---

## setAttribute(name, value)

### Description
Sets or modifies the value of a specified attribute on the element.

### Method
`setAttribute`

### Parameters
#### Path Parameters
- **name** (string) - The name of the attribute whose value is to be set.
- **value** (string) - The value to assign to the attribute.

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/setAttribute

---

## removeAttribute(name)

### Description
Removes a specified attribute from the element.

### Method
`removeAttribute`

### Parameters
#### Path Parameters
- **name** (string) - The name of the attribute to remove.

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/removeAttribute

---

## hasAttribute(name)

### Description
Returns a boolean indicating whether the element has the specified attribute.

### Method
`hasAttribute`

### Parameters
#### Path Parameters
- **name** (string) - The name of the attribute to check for.

### Returns
`boolean`

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/hasAttribute

---

## hasAttributes()

### Description
Returns a boolean value indicating whether the current element has any attributes or not.

### Method
`hasAttributes`

### Parameters
None

### Returns
`boolean`

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/hasAttributes

---

## getAttributeNames()

### Description
Returns the attribute names of the element as an Array of strings.

### Method
`getAttributeNames`

### Parameters
None

### Returns
`Array`

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttributeNames

---

## getAttributeNode(name)

### Description
Returns the attribute node with the specified name.

### Method
`getAttributeNode`

### Parameters
#### Path Parameters
- **name** (string) - The name of the attribute node to retrieve.

### Returns
`*`

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttributeNode

---

## setAttributeNode(newAttr)

### Description
Adds or replaces a specified attribute node in the element.

### Method
`setAttributeNode`

### Parameters
#### Path Parameters
- **newAttr** (*) - The attribute node to add or replace.

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/setAttributeNode

---

## removeAttributeNode(oldAttr)

### Description
Removes a specified attribute node from the element.

### Method
`removeAttributeNode`

### Parameters
#### Path Parameters
- **oldAttr** (*) - The attribute node to remove.

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/removeAttributeNode
```

---

### Render sp-menu-item Components in HTML

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-menu-item

Demonstrates how to render basic, selected, and disabled menu items using the sp-menu-item component. This is a fundamental usage example for creating menu options.

```html
<sp-menu-item>Chicago</sp-menu-item>
<sp-menu-item selected>New York City</sp-menu-item>
<sp-menu-item disabled>St. Louis</sp-menu-item>
```

---

### Element Attribute Management

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLSelectElement

Methods for getting, setting, and removing element attributes.

```APIDOC
## getAttribute(name)

### Description
Gets the value of a specified attribute on the element.

### Method
`getAttribute`

### Parameters
#### Path Parameters
- **name** (string) - Name of the attribute whose value you want to get.

### Returns
`string`

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttribute
```

```APIDOC
## setAttribute(name, value)

### Description
Sets the value of a specified attribute on the element.

### Method
`setAttribute`

### Parameters
#### Path Parameters
- **name** (string) - Name of the attribute whose value is to be set.
- **value** (string) - Value to assign to the attribute.

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/setAttribute
```

```APIDOC
## removeAttribute(name)

### Description
Removes a specified attribute from the element.

### Method
`removeAttribute`

### Parameters
#### Path Parameters
- **name** (string) - Description not available

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/removeAttribute
```

```APIDOC
## hasAttribute(name)

### Description
Checks if the element has a specified attribute.

### Method
`hasAttribute`

### Parameters
#### Path Parameters
- **name** (string) - Description not available

### Returns
`boolean`

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/hasAttribute
```

```APIDOC
## hasAttributes()

### Description
Returns a boolean value indicating whether the current element has any attributes or not.

### Method
`hasAttributes`

### Returns
`boolean`

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/hasAttributes
```

```APIDOC
## getAttributeNames()

### Description
Returns the attribute names of the element as an Array of strings.

### Method
`getAttributeNames`

### Returns
`Array`

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttributeNames
```

```APIDOC
## getAttributeNode(name)

### Description
Gets the attribute node with the specified name.

### Method
`getAttributeNode`

### Parameters
#### Path Parameters
- **name** (string) - Description not available

### Returns
`*`

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttributeNode
```

```APIDOC
## setAttributeNode(newAttr)

### Description
Sets a new attribute node on the element.

### Method
`setAttributeNode`

### Parameters
#### Path Parameters
- **newAttr** (*) - Description not available

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/setAttributeNode
```

```APIDOC
## removeAttributeNode(oldAttr)

### Description
Removes an attribute node from the element.

### Method
`removeAttributeNode`

### Parameters
#### Path Parameters
- **oldAttr** (*) - Description not available

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/removeAttributeNode
```

---

### Access UXP Shell API

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Import the `shell` object from the `uxp` module to access functionalities for interacting with the operating system's shell.

```javascript
const { shell } = require("uxp");
```

---

### Attribute Manipulation

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHeadElement

Methods for getting, setting, and removing attributes from an element.

````APIDOC
## getAttribute(name)

### Description
Retrieves the value of a specified attribute from the element.

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
// const attributeValue = element.getAttribute('data-custom');
````

### Response

#### Success Response (200)

Returns the attribute's value as a string, or null if the attribute is not set.

#### Response Example

N/A

## setAttribute(name, value)

### Description

Sets or updates the value of a specified attribute on the element.

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
// element.setAttribute('class', 'new-class');
```

### Response

#### Success Response (200)

N/A

#### Response Example

N/A

## removeAttribute(name)

### Description

Removes a specified attribute from the element.

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
// element.removeAttribute('disabled');
```

### Response

#### Success Response (200)

N/A

#### Response Example

N/A

## hasAttribute(name)

### Description

Checks if the element has a specified attribute.

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
// const hasDataAttribute = element.hasAttribute('data-id');
```

### Response

#### Success Response (200)

Returns `true` if the attribute exists, `false` otherwise.

#### Response Example

N/A

## hasAttributes()

### Description

Returns a boolean value indicating whether the current element has any attributes or not.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

N/A

### Request Example

```javascript
// const hasAnyAttributes = element.hasAttributes();
```

### Response

#### Success Response (200)

Returns `true` if the element has at least one attribute, `false` otherwise.

#### Response Example

N/A

## getAttributeNames()

### Description

Returns the attribute names of the element as an Array of strings.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

N/A

### Request Example

```javascript
// const attributeNames = element.getAttributeNames();
```

### Response

#### Success Response (200)

Returns an Array of strings, where each string is an attribute name.

#### Response Example

```json
{
	"example": ["id", "class", "data-custom"]
}
```

## getAttributeNode(name)

### Description

Retrieves the attribute node with the specified name.

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
// const attributeNode = element.getAttributeNode('style');
```

### Response

#### Success Response (200)

Returns the attribute node, or null if not found.

#### Response Example

N/A

## setAttributeNode(newAttr)

### Description

Adds a new attribute node to the element. If an attribute with the same name already exists, it is replaced.

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
// const newAttribute = document.createAttribute('data-info');
// newAttribute.value = 'some value';
// element.setAttributeNode(newAttribute);
```

### Response

#### Success Response (200)

Returns the attribute node that was added or replaced.

#### Response Example

N/A

## removeAttributeNode(oldAttr)

### Description

Removes an attribute node from the element.

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
// const attributeToRemove = element.getAttributeNode('data-info');
// if (attributeToRemove) {
//   element.removeAttributeNode(attributeToRemove);
// }
```

### Response

#### Success Response (200)

Returns the removed attribute node.

#### Response Example

N/A

````

--------------------------------

### Attribute Manipulation

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLButtonElement

Methods for getting, setting, and removing attributes from an element.

```APIDOC
## getAttribute(name)

### Description
Returns the value of a specified attribute on the element.

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
// const className = element.getAttribute('class');
````

### Response

#### Success Response (200)

- **Attribute Value** (`string`) - The value of the attribute, or `null` if the attribute is not set.

#### Response Example

```json
{
	"value": "some-class"
}
```

## setAttribute(name, value)

### Description

Sets or changes the value of a specified attribute on the element.

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
// element.setAttribute('id', 'new-id');
```

### Response

#### Success Response (200)

[No explicit return value, behavior is side-effectual]

#### Response Example

[None]

## removeAttribute(name)

### Description

Removes a specified attribute from the element.

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
// element.removeAttribute('data-custom');
```

### Response

#### Success Response (200)

[No explicit return value, behavior is side-effectual]

#### Response Example

[None]

## hasAttribute(name)

### Description

Returns a boolean indicating whether the element has the specified attribute.

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
// const hasId = element.hasAttribute('id');
```

### Response

#### Success Response (200)

- **Has Attribute** (`boolean`) - `true` if the attribute exists, `false` otherwise.

#### Response Example

```json
{
	"hasAttribute": true
}
```

## hasAttributes()

### Description

Returns a boolean value indicating whether the current element has any attributes or not.

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
// const hasAnyAttributes = element.hasAttributes();
```

### Response

#### Success Response (200)

- **Has Attributes** (`boolean`) - `true` if the element has one or more attributes, `false` otherwise.

#### Response Example

```json
{
	"hasAttributes": true
}
```

## getAttributeNames()

### Description

Returns the attribute names of the element as an Array of strings.

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
// const attributeNames = element.getAttributeNames();
```

### Response

#### Success Response (200)

- **Attribute Names** (`Array<string>`) - An array containing the names of all attributes of the element.

#### Response Example

```json
{
	"attributeNames": ["id", "class", "data-custom"]
}
```

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
