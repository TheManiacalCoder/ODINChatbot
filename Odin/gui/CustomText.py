import tkinter as tk
import tkinter.filedialog as filedialog
import pyperclip
import sys

class CustomText(tk.Text):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self.configure(
            bg="#282828",
            fg="#FFFFFF",
            insertbackground="#FFFFFF",
            padx=0,
            pady=0,
            highlightthickness=0,
            relief='flat',
            wrap='word'
        )

        # Configure tags for consistent formatting
        self.tag_configure(
            "user",
            background="#343434",
            foreground="#FFFFFF",
            font=("Segoe UI", 12),
            lmargin1=50,
            lmargin2=50,
            rmargin=100,
            wrap='word',
            spacing1=10,
            spacing2=0,
            spacing3=10
        )

        self.tag_configure(
            "assistant",
            background="#1E1E1E",
            foreground="#FFFFFF",
            font=("Segoe UI", 12),
            lmargin1=50,
            lmargin2=50,
            rmargin=100,
            wrap='word',
            spacing1=10,
            spacing2=0,
            spacing3=10
        )

        self.tag_configure(
            "code",
            background="#000000",
            foreground="#FFFFFF",
            font=("Courier New", 12),
            lmargin1=50,
            lmargin2=50,
            rmargin=100,
            wrap='word',
            spacing1=10,
            spacing2=0,
            spacing3=10
        )

        self.tag_configure(
            "buttons",
            background="#000000",
            foreground="#0078D7",
            font=("Courier New", 12, "underline"),  # Same font as code with underline
            lmargin1=50,
            lmargin2=50,
            rmargin=100,
            wrap='none',
            spacing1=10,
            spacing2=0,
            spacing3=10
        )

        self.tag_bind("buttons", "<Button-1>", self.handle_button_click)
        self.tag_bind("buttons", "<Enter>", self.on_button_enter)
        self.tag_bind("buttons", "<Leave>", self.on_button_leave)

    def handle_button_click(self, event):
        index = self.index("@%s,%s" % (event.x, event.y))
        print(f"Clicked at index: {index}")  # Debug: Print the clicked index

        # Find the nearest code block adjacent to the buttons
        code_start, code_end = self.tag_prevrange("code", index)
        if code_start and code_end:
            code_content = self.get(code_start, code_end).strip()
            print(f"Code content: {code_content}")  # Debug: Print the code content

            # Check which button was clicked
            clicked_word = self.get(index, index + " wordend")
            print(f"Clicked word: {clicked_word}")  # Debug: Print the clicked word
            if "Copy" in clicked_word:
                self.copy_code_to_clipboard(code_content)
            elif "Save" in clicked_word:
                self.save_code_to_file(code_content)
        else:
            print("Error: Could not find the adjacent code block.")  # Debug: Error message

    def copy_code_to_clipboard(self, code_content):
        pyperclip.copy(code_content)
        print("Codeblock Copied!")  # Print to cmd

    def save_code_to_file(self, code_content):
        file_path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python Files", "*.py"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, "w") as file:
                file.write(code_content)
            print("Codeblock Saved!")  # Print to cmd

    def insert_code(self, code, language=None):
        """Insert code content without backticks."""
        self.insert('end', code + "\n", "code")
        self.insert('end', "[Copy] [Save]\n", ("buttons",))
        self.mark_set("insert", "end-2c")  # Move the insert mark to the end of the code block

    def on_button_enter(self, event):
        self.config(cursor="hand2")

    def on_button_leave(self, event):
        self.config(cursor="")